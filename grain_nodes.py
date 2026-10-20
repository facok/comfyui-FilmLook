# -*- coding: utf-8 -*-
"""颗粒节点：FilmLookGrain（基于 silvergrain 的物理胶片颗粒）。"""

import inspect
import logging
import math

import numpy as np
import torch

from . import film_core

logger = logging.getLogger("FilmLook")

GRAIN_LEVELS = {
    "极细 (ISO 100)": 0.06,
    "细腻 (ISO 200)": 0.08,
    "标准 (ISO 400)": 0.11,
    "明显 (ISO 800)": 0.15,
    "粗糙 (ISO 1600)": 0.21,
    "极粗 (ISO 3200)": 0.28,
}

GRAIN_TYPES = {
    "亮度颗粒 (保色彩)": "luminance",
    "彩色颗粒 (胶片感更强)": "rgb",
}

QUALITY_LEVELS = {
    "速度优先": 100,
    "均衡": 200,
    "画质优先": 400,
}

_RENDERER_CACHE = {}

_SILVERGRAIN_HELP = (
    "胶片颗粒功能需要 silvergrain 库，但当前 Python 环境未安装。\n"
    "请在 ComfyUI 的 Python 环境中运行：pip install silvergrain[gpu]\n"
    "注意：需要 NVIDIA CUDA 显卡才有实用速度（CPU 渲染 1024x1024 约需 159 秒）；"
    "silvergrain 采用 AGPLv3 开源协议，商用前请确认合规。"
)


def _import_silvergrain():
    try:
        from silvergrain import FilmGrainRenderer
    except ImportError as e:
        raise RuntimeError(_SILVERGRAIN_HELP) from e
    return FilmGrainRenderer


def _get_renderer(grain_radius, n_monte_carlo):
    key = (grain_radius, n_monte_carlo)
    if key not in _RENDERER_CACHE:
        FilmGrainRenderer = _import_silvergrain()
        # sigma_filter 取偏锐利的 0.5：0.8 的抗锯齿会明显软化画面细节
        kwargs = {"grain_radius": grain_radius, "n_monte_carlo": n_monte_carlo,
                  "grain_sigma": 0.03, "sigma_filter": 0.5, "device": "auto"}
        params = inspect.signature(FilmGrainRenderer).parameters
        if not any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
            kwargs = {k: v for k, v in kwargs.items() if k in params}
        _RENDERER_CACHE[key] = FilmGrainRenderer(**kwargs)
    return _RENDERER_CACHE[key]


def _unsharp_mask(image, amount):
    """亮度域反锐化蒙版：补偿颗粒渲染的光学柔化，只锐化明暗、不放大彩色噪点。"""
    if amount <= 0.0:
        return image
    rgb = image[..., :3]
    lum = film_core.luminance(rgb)
    lum = lum + amount * (lum - film_core._gaussian_blur(lum, 1.2))
    out = rgb + (lum - film_core.luminance(rgb))
    if image.shape[-1] == 4:
        out = torch.cat([out, image[..., 3:4]], dim=-1)
    return out.clamp(0.0, 1.0)


_LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


def _render_frame(renderer, frame_np, mode):
    """float32 直达渲染内核（render_single_channel），绕开 process_image 的 uint8 量化。
    luminance 模式在 Y 上渲染后把亮度差回加 RGB，避免 silvergrain 自带的 uint8 YUV 往返。"""
    if mode == "rgb":
        ch = [renderer.render_single_channel(np.ascontiguousarray(frame_np[:, :, c]),
                                             zoom=1.0, output_size=None) for c in range(3)]
        return np.stack(ch, axis=2)
    y = np.ascontiguousarray(frame_np @ _LUMA)
    y_new = renderer.render_single_channel(y, zoom=1.0, output_size=None)
    return (frame_np + (y_new - y)[..., None]).clip(0.0, 1.0)


def apply_grain(image, grain_level, grain_type, strength, quality, seed, sharpen=0.3):
    """逐帧渲染物理胶片颗粒。全程 float32（无 8-bit 量化）；颗粒半径按画面尺寸
    相对 1024² 缩放，保证同一档位在不同分辨率下视觉粗细一致；渲染内核自带逐帧
    随机采样，满足颗粒时序独立（seed 参数用于 ComfyUI 缓存刷新）。"""
    radius_base = GRAIN_LEVELS[grain_level]
    mode = GRAIN_TYPES[grain_type]
    n_mc = QUALITY_LEVELS[quality]
    if not torch.cuda.is_available():
        logger.warning("FilmLook: 未检测到 CUDA，silvergrain 使用 CPU 渲染会很慢（1024x1024 约 159 秒）。")
    frames = []
    for i in range(image.shape[0]):
        frame = image[i, :, :, :3]
        alpha = image[i, :, :, 3:4] if image.shape[-1] == 4 else None
        arr = frame.clamp(0.0, 1.0).cpu().numpy().astype(np.float32)
        h, w = arr.shape[:2]
        radius = round(radius_base * math.sqrt(h * w) / 1024.0, 4)
        renderer = _get_renderer(radius, n_mc)
        rendered = _render_frame(renderer, arr, mode)
        out_np = arr + (rendered - arr) * float(strength)
        out = torch.from_numpy(out_np.astype(np.float32))
        if alpha is not None:
            out = torch.cat([out, alpha.cpu().float()], dim=-1)
        frames.append(out)
    out = torch.stack(frames, dim=0).to(image.device)
    return _unsharp_mask(out, float(sharpen))


class FilmLookGrain:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image": ("IMAGE",),
            "grain_level": (list(GRAIN_LEVELS.keys()), {"default": "标准 (ISO 400)",
                                                        "tooltip": "颗粒粗细：感光度越高颗粒越粗"}),
            "grain_type": (list(GRAIN_TYPES.keys()), {"default": "亮度颗粒 (保色彩)",
                                                      "tooltip": "亮度颗粒只影响明暗、不改变颜色；彩色颗粒每通道独立，更有胶片味但会轻微偏色"}),
            "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
                                   "tooltip": "颗粒强度：0 为原图，1 为完整颗粒"}),
            "quality": (list(QUALITY_LEVELS.keys()), {"default": "均衡",
                                                      "tooltip": "渲染质量：越高颗粒越干净细腻，速度越慢"}),
            "sharpen": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.05,
                                  "tooltip": "锐度补偿：颗粒渲染的光学特性会轻微柔化画面，此项加回线条锐度；0 为完全关闭"}),
            "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF,
                             "tooltip": "随机种子：改变颗粒图案；同一批次每帧自动使用不同随机"}),
        }}

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("图像",)
    FUNCTION = "execute"
    CATEGORY = "FilmLook/颗粒"

    def execute(self, image, grain_level, grain_type, strength, quality, sharpen, seed):
        return (apply_grain(image, grain_level, grain_type, strength, quality, seed, sharpen),)
