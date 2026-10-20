# -*- coding: utf-8 -*-
"""一键节点：FilmLookOneClick（调色 + 光晕 + 颗粒一次完成）。"""

from . import film_core
from .film_stocks import FILM_STOCKS, PRESET_LABELS
from .grain_nodes import GRAIN_LEVELS, apply_grain
from .halation_nodes import RADIUS_PRESETS


class FilmLookOneClick:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image": ("IMAGE",),
            "preset": (PRESET_LABELS, {"default": PRESET_LABELS[0],
                                       "tooltip": "选择胶片风格，与“胶片调色”节点相同"}),
            "grain_level": (list(GRAIN_LEVELS.keys()), {"default": "标准 (ISO 400)",
                                                        "tooltip": "颗粒粗细：感光度越高颗粒越粗"}),
            "strength": ("FLOAT", {"default": 0.85, "min": 0.0, "max": 1.0, "step": 0.01,
                                   "tooltip": "整体强度：同时作用于调色与颗粒"}),
            "halation_strength": ("FLOAT", {"default": 0.4, "min": 0.0, "max": 1.0, "step": 0.01,
                                            "tooltip": "光晕强度：亮部周围红橙色雾圈的明显程度"}),
            "auto_exposure": ("BOOLEAN", {"default": True,
                                          "tooltip": "智能曝光：自动校正整体亮度（含高光保护）；画面过曝时可关闭"}),
            "preserve_luminance": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05,
                                             "tooltip": "亮度保持：0 允许胶片曲线自由改变亮度，1 输出亮度完全贴回原图；遇到过曝时调高此值"}),
            "style_boost": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 2.0, "step": 0.05,
                                      "tooltip": "风格浓度：1 为忠实胶片效果，大于 1 把风格放大到超出真实胶片"}),
            "diffusion": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.05,
                                    "tooltip": "柔光强度：高光弥散的柔焦感（黑柔镜效果），0 关闭"}),
            "vignette": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.05,
                                   "tooltip": "暗角强度：四角向中心压暗，0 关闭"}),
            "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF,
                             "tooltip": "随机种子：改变颗粒图案；同一批次每帧自动使用不同随机"}),
        }}

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("图像",)
    FUNCTION = "execute"
    CATEGORY = "FilmLook"

    def execute(self, image, preset, grain_level, strength, halation_strength,
                auto_exposure, preserve_luminance, style_boost, diffusion, vignette, seed):
        out = film_core.apply_film_color(image, FILM_STOCKS[preset], strength=strength,
                                         auto_exposure=auto_exposure,
                                         preserve_luminance=preserve_luminance,
                                         style_boost=style_boost)
        out = film_core.apply_diffusion(out, strength=diffusion)
        out = film_core.apply_halation(out, strength=halation_strength,
                                       scales=RADIUS_PRESETS["标准"], auto_threshold=True)
        out = film_core.apply_vignette(out, strength=vignette)
        out = apply_grain(out, grain_level, "亮度颗粒 (保色彩)", strength, "均衡", seed)
        return (out,)
