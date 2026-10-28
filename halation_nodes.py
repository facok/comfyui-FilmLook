# -*- coding: utf-8 -*-
"""光学节点：FilmLookHalation（胶片高光红橙光晕）。"""

from . import film_core

RADIUS_PRESETS = {
    "轻微": (0.004,),
    "标准": (0.01, 0.02),
    "明显": (0.02, 0.04),
    "夸张": (0.035, 0.07, 0.14),
}


class FilmLookHalation:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image": ("IMAGE",),
            "strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01,
                                   "tooltip": "光晕强度：0 无光晕，1 最明显"}),
            "radius": (list(RADIUS_PRESETS.keys()), {"default": "标准",
                                                     "tooltip": "光晕范围：越大，亮部周围的红橙色雾圈越宽"}),
            "auto_threshold": ("BOOLEAN", {"default": True,
                                           "tooltip": "自动检测高光：开启后自动找出画面最亮的区域产生光晕"}),
            "threshold": ("FLOAT", {"default": 0.85, "min": 0.0, "max": 1.0, "step": 0.01,
                                    "tooltip": "手动高光阈值：仅在关闭自动检测时生效，数值越低越多区域产生光晕（以画面显示亮度为准）"}),
        }}

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("图像",)
    FUNCTION = "execute"
    CATEGORY = "FilmLook/光学"

    def execute(self, image, strength, radius, auto_threshold, threshold):
        image = film_core.sanitize(image)
        out = film_core.apply_halation(image, strength=strength, scales=film_core.pick("光晕范围", radius, RADIUS_PRESETS),
                                       auto_threshold=auto_threshold, threshold=threshold)
        return (out,)
