# -*- coding: utf-8 -*-
"""胶片预设表：一个预设 = 一条 3D LUT + 少量元数据。

设计变更（v2）：预设不再用参数化模型（curve / crosstalk / split_tone / sat_scale）
去拟合 LUT，而是**直接把 LUT 应用到画面**。

原因：拟合会丢性格。实测用 12 参数模型逼近参考 LUT，765 色板平均 |Δ| 0.04~0.08、
95 分位最高 0.25；Vision3 500T 的天空色度被从 LUT 的 0.235 放大到 0.559（2.4 倍），
"冷调"因此变成了最暖最浓的预设。直接查表则 100% 复现参考 LUT，且跑得比拟合管线更快。

LUT 来源与许可（含 CC BY-SA 提示）见 `presets/LICENSE.md`。
"""

import os

from . import film_core

PRESET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "presets")

FILM_STOCKS = {
    "清新人像 (Portra 400)": {"lut": "portra_400.cube", "monochrome": False},
    "日系空气感 (Fuji 400H)": {"lut": "fuji_400h.cube", "monochrome": False},
    "暖阳金调 (Kodak Gold)": {"lut": "kodak_gold_200_warm.cube", "monochrome": False},
    "冷调电影感 (Vision3 500T)": {"lut": "vision3_500t.cube", "monochrome": False},
    "浓郁鲜艳 (Ektar)": {"lut": "ektar_100.cube", "monochrome": False},
    "黑白纪实 (Tri-X)": {"lut": "trix_400.cube", "monochrome": True},
    "自定义": {"lut": None, "monochrome": False},
}

PRESET_LABELS = list(FILM_STOCKS.keys())


def preset_lut(name):
    """解析预设对应的 LUT（走 film_core 的解析缓存）；"自定义" 返回 None。"""
    filename = FILM_STOCKS[name]["lut"]
    if not filename:
        return None
    return film_core.parse_cube(os.path.join(PRESET_DIR, filename))


def preset_entry(name):
    """给 apply_film_color 用的 stock 字典：元数据 + 已解析的 table。"""
    entry = dict(FILM_STOCKS[name])
    entry["table"] = preset_lut(name)
    return entry
