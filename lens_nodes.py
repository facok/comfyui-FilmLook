# -*- coding: utf-8 -*-
"""镜头节点：FilmLookLens（柔光 + 暗角，镜头光学质感）。"""

from . import film_core


class FilmLookLens:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image": ("IMAGE",),
            "diffusion": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.05,
                                    "tooltip": "柔光强度：高光向周围弥散的柔焦效果（黑柔镜感），0 关闭"}),
            "vignette": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.05,
                                   "tooltip": "暗角强度：四角向中心压暗，引导视线，0 关闭"}),
        }}

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("图像",)
    FUNCTION = "execute"
    CATEGORY = "FilmLook/光学"

    def execute(self, image, diffusion, vignette):
        out = film_core.apply_diffusion(image, strength=diffusion)
        out = film_core.apply_vignette(out, strength=vignette)
        return (out,)
