# -*- coding: utf-8 -*-
"""FilmLook：胶片质感套件（调色 / LUT / 光晕 / 颗粒 / 一键）。"""

from .color_nodes import FilmLookColor, FilmLookLUT
from .combo_nodes import FilmLookOneClick
from .grain_nodes import FilmLookGrain
from .halation_nodes import FilmLookHalation

NODE_CLASS_MAPPINGS = {
    "FilmLook_Color": FilmLookColor,
    "FilmLook_LUT": FilmLookLUT,
    "FilmLook_Halation": FilmLookHalation,
    "FilmLook_Grain": FilmLookGrain,
    "FilmLook_OneClick": FilmLookOneClick,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FilmLook_Color": "FilmLook 胶片调色",
    "FilmLook_LUT": "FilmLook LUT 应用",
    "FilmLook_Halation": "FilmLook 光晕 (Halation)",
    "FilmLook_Grain": "FilmLook 胶片颗粒",
    "FilmLook_OneClick": "FilmLook 一键胶片",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

print("[34mFilmLook: [92mLoaded custom nodes.[0m")
