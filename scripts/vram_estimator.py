import time
import logging
import socket
import json
import math
import os.path
import torch.cuda
import numpy as np
import scipy.linalg
import scipy.optimize
from collections import defaultdict
from hashlib import sha256
from PIL import Image
from logging.handlers import SysLogHandler
from modules import shared, script_callbacks, devices, scripts
from modules.processing import StableDiffusionProcessingTxt2Img, StableDiffusionProcessingImg2Img, Processed, process_images
import gradio as gr
import pandas as pd


DEFAULT_ARGS = {
    'sd_model': shared.sd_model,
    'prompt': 'postapocalyptic steampunk city, exploration, cinematic, realistic, hyper detailed, photorealistic maximum detail, volumetric light, (((focus))), wide-angle, (((brightly lit))), (((vegetation))), lightning, vines, destruction, devastation, wartorn, ruins',
    'sampler_name': 'Euler a',
    'batch_size': 1,
    'n_iter': 1,
    'steps': 1,
    'cfg_scale': 15.0,
    'width': 512,
    'height': 512,
    'restore_faces': False,
    'tiling': False,
    'do_not_save_samples': True,
    'do_not_save_grid': True,
    'negative_prompt': '(((blurry))), ((foggy)), (((dark))), ((monochrome)), sun, (((depth of field)))',
    'do_not_reload_embeddings': True
}


curve_txt2img = None
curve_img2img = None
stats_file = os.path.join(scripts.basedir(), "stats.json")


class VRAMCurve():
    """3d plane where x=pixels, y=batch_size, z=reserved_peak"""
    def __init__(self, data):
        arr = []
        batch_sizes_x = defaultdict(list)
        batch_sizes_y = defaultdict(list)
        for entry in data:
            size = math.sqrt(entry["pixels"])
            batch_size = entry["batch_size"]
            reserved = entry["reserved_peak"]

            arr.append([size, batch_size, reserved])
            batch_sizes_x[batch_size].append(size)
            batch_sizes_y[batch_size].append(reserved)

        data = np.array(arr)

        # VRAM usage seems to scale linearly within one batch size, but the rate
        # of increase differs between batch sizes with no real pattern
        self.batch_sizes = {}
        for k in batch_sizes_x.keys():
            z = np.polyfit(batch_sizes_x[k], batch_sizes_y[k], 3)
            f = np.poly1d(z)
            self.batch_sizes[k] = f

        # best-fit cubic curve
        # M = [ones(size(x)), x, y, x.^2, x.*y, y.^2, x.^3, x.^2.*y, x.*y.^2, y.^3]
        A = np.c_[np.ones(data.shape[0]), data[:,:2], data[:,0]**2, np.prod(data[:,:2], axis=1), \
                data[:,1]**2, data[:,0]**3, np.prod(np.c_[data[:,0]**2,data[:,1]],axis=1), \
                np.prod(np.c_[data[:,0],data[:,1]**2],axis=1), data[:,1]**3]
        self.C,_,_,_ = scipy.linalg.lstsq(A, data[:,2])

    def estimate(self, pixels, batch_size):
        x = math.sqrt(pixels)
        y = batch_size

        if batch_size in self.batch_sizes:
            return self.batch_sizes[batch_size](x)

        return np.dot([1, x, y,
                       x ** 2, x * y, y ** 2,
                       x ** 3, x ** 2 * y, x * y ** 2, y ** 3], self.C)


def load_curve():
    global curve_txt2img, curve_img2img
    if not os.path.isfile(stats_file):
        print("[VRAMEstimator] No stats available, run benchmark first")
        return None, None

    with open(stats_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "txt2img" not in data or "img2img" not in data:
        print("[VRAMEstimator] No stats available, run benchmark first")
        return None, None

    curve_txt2img = VRAMCurve(data["txt2img"])
    curve_img2img = VRAMCurve(data["img2img"])
    print("[VRAMEstimator] Loaded benchmark data.")
    return make_plots(data)


def get_memory_stats():
    devices.torch_gc()
    torch.cuda.reset_peak_memory_stats()
    shared.mem_mon.monitor()
    return {k: -(v//-(1024*1024)) for k, v in shared.mem_mon.stop().items()}


def run_benchmark(max_width, max_batch_count):
    global curve_txt2img, curve_img2img
    results = {}

    print("[VRAMEstimator] Starting benchmark...")
    mem_stats = get_memory_stats()
    base_active = mem_stats["active_peak"]
    base_reserved = mem_stats["reserved_peak"]
    base_used = mem_stats["total"] - mem_stats["free"]

    shared.state.begin()
    shared.state.job = "VRAM Estimator Benchmark"
    shared.state.job_count = max_batch_count * int((max_width - 256) / 64)

    for op in ["txt2img", "img2img"]:
        results[op] = []
        for b in range(1, max_batch_count+1):
            for i in range(256, max_width+64, 64):
                print(f"run benchmark: {op} {b}x{i}")
                devices.torch_gc()
                torch.cuda.reset_peak_memory_stats()
                shared.mem_mon.monitor()
                shared.state.begin()

                args = DEFAULT_ARGS.copy()
                args["batch_size"] = b
                args["width"] = i
                args["height"] = i

                if op == "txt2img":
                    p = StableDiffusionProcessingTxt2Img(**args)
                elif op == "img2img":
                    args["init_images"] = [Image.new("RGB", (512, 512))]
                    p = StableDiffusionProcessingImg2Img(**args)
                else:
                    print(f'unknown operation: {op}')
                    return 'error'

                t0 = time.time()
                try:
                    process_images(p)
                except Exception as e:
                    print(f'benchmark error: {e}')
                    return 'error'
                t1 = time.time()
                shared.state.end()

                its = args["steps"] * args["batch_size"] / (t1 - t0)

                mem_stats = {k: -(v//-(1024*1024)) for k, v in shared.mem_mon.stop().items()}
                results[op].append({
                    "width": args["width"],
                    "pixels": args["width"] * args["height"],
                    "batch_size": args["batch_size"],
                    "active_peak": mem_stats["active_peak"] - base_active,
                    "reserved_peak": mem_stats["reserved_peak"] - base_reserved,
                    "sys_peak": mem_stats["system_peak"],
                    "sys_total": mem_stats["total"],
                    "its": round(its, 2)
                })

                with open(stats_file, "w", encoding="utf-8") as f:
                    json.dump(results, f)

                del p
                del args

    curve_txt2img = VRAMCurve(results["txt2img"])
    curve_img2img = VRAMCurve(results["img2img"])

    shared.state.end()
    print("[VRAMEstimator] Benchmark finished.")

    results = make_plots(results)
    results.append("<span>Benchmark finished.</span>")
    return results


def make_plots(results):
    dfs = []

    for op in ["txt2img", "img2img"]:
        x = []
        y = []
        z = []
        w = []

        for result in results[op]:
            x.append(result["pixels"])
            y.append(result["batch_size"])
            z.append(result["reserved_peak"])
            w.append(result["its"])

        df = pd.DataFrame({"pixels": x, "reserved_peak": z, "batch_size": y})
        df2 = pd.DataFrame({"pixels": x, "its": w, "batch_size": y})
        dfs.extend([df, df2])

    return dfs


def on_ui_tabs():
    load_curve()

    with gr.Blocks(analytics_enabled=False) as vram_estimator_tab:
        with gr.Row():
            plot = gr.LinePlot(title="txt2img Reserved VRAM", x="pixels", y="reserved_peak", color="batch_size", width=400, height=400, tooltip=["pixels", "reserved_peak", "batch_size"])
            plot2 = gr.LinePlot(title="txt2img it/s", x="pixels", y="its", color="batch_size", width=400, height=400, tooltip=["pixels", "its", "batch_size"])
        with gr.Row():
            plot3 = gr.LinePlot(title="img2img Reserved VRAM", x="pixels", y="reserved_peak", color="batch_size", width=400, height=400, tooltip=["pixels", "reserved_peak", "batch_size"])
            plot4 = gr.LinePlot(title="img2img it/s", x="pixels", y="its", color="batch_size", width=400, height=400, tooltip=["pixels", "its", "batch_size"])
        with gr.Row():
            width = gr.Slider(minimum=256, maximum=2048, step=64, label="Max Image Size", value=1024)
            batch_count = gr.Slider(minimum=1, maximum=16, step=1, label="Max Batch Count", value=8)
            with gr.Column():
                bench_run_btn = gr.Button("Run benchmark", variant="primary").style(full_width=False)
                load_results_button = gr.Button("Load results").style(full_width=False)
                with gr.Row():
                    status = gr.HTML("")

        bench_run_btn.click(run_benchmark, inputs=[width, batch_count], outputs=[plot, plot2, plot3, plot4, status])
        load_results_button.click(load_curve, inputs=[], outputs=[plot, plot2, plot3, plot4])

    return [(vram_estimator_tab, "VRAM Estimator", "vram_estimator_tab")]


heatmap = [
    [-1.0, (0.2, 0.2, 1.0)],
    [ 0.0, (0.2, 0.2, 1.0)],
    [ 0.2, (0.0, 1.0, 1.0)],
    [ 0.4, (0.0, 1.0, 0.0)],
    [ 0.6, (1.0, 1.0, 0.0)],
    [ 0.8, (1.0, 0.0, 0.0)],
    [ 1.0, (1.0, 0.0, 0.8)],
    [ 2.0, (1.0, 0.0, 0.8)],
]


def gaussian(x, a, b, c, d=0):
    return a * math.exp(-(x - b)**2 / (2 * c**2)) + d

def get_color(x, map=[], spread=1):
    x = min(1.0, max(0, x))
    r = min(1.0, sum([gaussian(x, p[1][0], p[0], 1.0 / (spread * len(map))) for p in map]))
    g = min(1.0, sum([gaussian(x, p[1][1], p[0], 1.0 / (spread * len(map))) for p in map]))
    b = min(1.0, sum([gaussian(x, p[1][2], p[0], 1.0 / (spread * len(map))) for p in map]))
    return 'rgb(%.2f, %.2f, %.2f)' % (r * 255, g * 255, b * 255)


def make_span(reserved_vram_estimate):
    if shared.state.job_count != 0:
        return '<div style="padding: 10px"><span>(Currently generating...)</span></div>'

    mem_stats = get_memory_stats()

    # VRAM estimate is taken after used system VRAM is subtracted
    base_rest = mem_stats["total"] - mem_stats["free"]
    estimate_total = base_rest + reserved_vram_estimate * 1.1 # buffer

    formatted_est = f"{round(estimate_total, 2):.2f} MB"
    formatted_total = f"{mem_stats['total']} MB"
    percent_usage = estimate_total / mem_stats['total']
    color = get_color(percent_usage, map=heatmap, spread=1.5)

    return f'''
    <div style="padding: 10px">
      <div>Estimated VRAM usage: <span style="color: {color}">{formatted_est} / {formatted_total} ({percent_usage * 100:.2f}%)</span></div>
      <div>({base_rest} MB system + {reserved_vram_estimate:.2f} MB used)</div>
    </div>
    '''


def estimate_vram_txt2img(width, height, batch_size, enable_hr, hr_scale, hr_resize_x, hr_resize_y):
    global curve_txt2img
    if not curve_txt2img:
        return "<span>(No stats yet, run benchmark in VRAM Estimator tab)</span>"

    final_width = width
    final_height = height

    if enable_hr:
        if hr_resize_x == 0 and hr_resize_y == 0:
            final_width = width * hr_scale
            final_height = height * hr_scale
        else:
            if hr_resize_y == 0:
                final_width = hr_resize_x
                final_height = hr_resize_x * height // width
            elif hr_resize_x == 0:
                final_width = hr_resize_y * width // height
                final_height = hr_resize_y
            else:
                src_ratio = width / height
                dst_ratio = hr_resize_x / hr_resize_y

                if src_ratio < dst_ratio:
                    final_width = hr_resize_x
                    final_height = hr_resize_x * height // width
                else:
                    final_width = hr_resize_y * width // height
                    final_height = hr_resize_y

    vram_estimate = curve_txt2img.estimate(final_width * final_height, batch_size)
    return make_span(vram_estimate)


def estimate_vram_img2img(width, height, batch_size):
    global curve_img2img
    if not curve_img2img:
        return "<span>(No stats yet, run benchmark in VRAM Estimator tab)</span>"

    vram_estimate = curve_img2img.estimate(width * height, batch_size)
    return make_span(vram_estimate)


class Script(scripts.Script):
    def title(self):
        return "VRAM Usage Estimator"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        with gr.Group():
            with gr.Row():
                vram_status = gr.HTML()

        if is_img2img:
            inputs = [self.i2i_width, self.i2i_height, self.i2i_batch_size]
            fn = estimate_vram_img2img
        else:
            inputs = [self.t2i_width, self.t2i_height, self.t2i_batch_size, self.t2i_enable_hr, self.t2i_hr_scale, self.t2i_hr_resize_x, self.t2i_hr_resize_y]
            fn = estimate_vram_txt2img

        for input in inputs:
            input.change(
                fn=fn,
                inputs=inputs,
                outputs=[vram_status],
                show_progress=False,
            )

        return [vram_status]

    def after_component(self, component, **kwargs):
        elem_id = kwargs.get("elem_id")

        if   elem_id == "txt2img_width":        self.t2i_width = component
        elif elem_id == "txt2img_height":       self.t2i_height = component
        elif elem_id == "txt2img_batch_size":   self.t2i_batch_size = component
        elif elem_id == "txt2img_enable_hr":    self.t2i_enable_hr = component
        elif elem_id == "txt2img_hr_scale":     self.t2i_hr_scale = component
        elif elem_id == "txt2img_hr_resize_x":  self.t2i_hr_resize_x = component
        elif elem_id == "txt2img_hr_resize_y":  self.t2i_hr_resize_y = component
        elif elem_id == "img2img_width":        self.i2i_width = component
        elif elem_id == "img2img_height":       self.i2i_height = component
        elif elem_id == "img2img_batch_size":   self.i2i_batch_size = component


script_callbacks.on_ui_tabs(on_ui_tabs)
