# -*- coding: utf-8 -*-
"""色彩节点：FilmLookColor（胶片调色）与 FilmLookLUT（.cube 应用）。"""

import os

from . import film_core
from .film_stocks import FILM_STOCKS, PRESET_LABELS

LUTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "luts")
CUSTOM_PATH_ENTRY = "自定义路径"

_PRESET_TOOLTIP = (
    "选择胶片风格："
    "清新人像 (Portra 400)——低反差暖调、肤色柔和，适合人像与日常；"
    "日系空气感 (Fuji 400H)——轻盈通透、暗部微青，适合小清新与街拍；"
    "暖阳金调 (Kodak Gold)——金黄暖调，适合阳光、旅行与复古；"
    "冷调电影感 (Vision3 500T)——青橙对比、冷调暗部，适合夜景与都市；"
    "浓郁鲜艳 (Ektar)——高饱和高反差，适合风光与静物；"
    "黑白纪实 (Tri-X)——高反差黑白，适合纪实与特写；"
    "自定义——中性曲线，只响应下方手动参数"
)


class FilmLookColor:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image": ("IMAGE",),
            "preset": (PRESET_LABELS, {"default": PRESET_LABELS[0], "tooltip": _PRESET_TOOLTIP}),
            "strength": ("FLOAT", {"default": 0.85, "min": 0.0, "max": 1.0, "step": 0.01,
                                   "tooltip": "效果强度：0 为原图，1 为完整胶片效果"}),
            "auto_exposure": ("BOOLEAN", {"default": True,
                                          "tooltip": "智能曝光：自动把整体亮度校正到胶片友好的区间"}),
            "brightness": ("FLOAT", {"default": 0.0, "min": -2.0, "max": 2.0, "step": 0.05,
                                     "tooltip": "亮度：+1 约提亮一档，-1 约压暗一档"}),
            "contrast": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01,
                                   "tooltip": "对比度：1 保持不变，越大明暗反差越强"}),
            "saturation": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01,
                                     "tooltip": "饱和度：1 保持不变；提升时优先增强灰暗颜色，肤色更自然"}),
            "temperature": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01,
                                      "tooltip": "色温：负值偏冷（蓝），正值偏暖（黄）；"
                                                 "满行程约相当于 ±65 mired（等效色温 6500K → 4600K / 11600K）"}),
            "tint": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01,
                               "tooltip": "色调：负值偏绿，正值偏品红；与色温同量级，只改色相不改亮度"}),
            "preserve_luminance": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05,
                                             "tooltip": "亮度保持：0 允许胶片曲线自由改变亮度，1 输出亮度完全贴回原图（只保留色彩与颗粒感）；遇到过曝时调高此值"}),
            "style_boost": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 2.0, "step": 0.05,
                                      "tooltip": "风格浓度：1 为忠实胶片效果，大于 1 把风格放大到超出真实胶片"}),
        }}

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("图像",)
    FUNCTION = "execute"
    CATEGORY = "FilmLook/色彩"

    def execute(self, image, preset, strength, auto_exposure, brightness, contrast,
                saturation, temperature, tint, preserve_luminance, style_boost):
        out = film_core.apply_film_color(
            image, FILM_STOCKS[preset], strength=strength, auto_exposure=auto_exposure,
            brightness=brightness, contrast=contrast, saturation=saturation,
            temperature=temperature, tint=tint, preserve_luminance=preserve_luminance,
            style_boost=style_boost)
        return (out,)


def _list_luts():
    try:
        files = sorted(f for f in os.listdir(LUTS_DIR) if f.lower().endswith((".cube", ".png")))
    except OSError:
        files = []
    return files + [CUSTOM_PATH_ENTRY]


def _parse_lut(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".cube":
        return film_core.parse_cube(path)
    if ext == ".png":
        return film_core.parse_haldclut(path)
    raise RuntimeError(f"不支持的 LUT 格式“{ext}”，目前支持 .cube 和 HaldCLUT .png")


class FilmLookLUT:
    @classmethod
    def INPUT_TYPES(cls):
        luts = _list_luts()
        return {
            "required": {
                "image": ("IMAGE",),
                "lut": (luts, {"default": luts[0],
                               "tooltip": "选择 LUT：把 .cube 或 HaldCLUT .png 文件放进插件的 luts 文件夹即可出现在这里；选择“自定义路径”则使用下方手动路径"}),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
                                       "tooltip": "效果强度：0 为原图，1 为完整 LUT 效果"}),
            },
            "optional": {
                "custom_path": ("STRING", {"default": "",
                                           "tooltip": "自定义 .cube / HaldCLUT .png 文件完整路径，仅在选择“自定义路径”时生效"}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("图像",)
    FUNCTION = "execute"
    CATEGORY = "FilmLook/色彩"

    def execute(self, image, lut, strength, custom_path=""):
        if lut == CUSTOM_PATH_ENTRY:
            path = (custom_path or "").strip().strip('"')
            if not path:
                raise RuntimeError("已选择“自定义路径”，请在 custom_path 中填写 .cube 或 HaldCLUT .png 文件的完整路径。")
        else:
            path = os.path.join(LUTS_DIR, lut)
            # combo 值不可信：确认解析后的路径仍在 luts 目录内
            if os.path.dirname(os.path.abspath(path)) != os.path.abspath(LUTS_DIR):
                raise RuntimeError(f"非法的 LUT 文件名：{lut}")
        parsed = _parse_lut(path)
        return (film_core.apply_cube(image, parsed, strength),)
