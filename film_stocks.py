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

# 每个预设的**实测性格**（不是照胶片名声写的）。数字来自 tests/_preset_metrics.py，
# 由 tests/preset_character_test.py 断言"描述与实测一致"——换 LUT 或改管线会让测试失败，
# 从而逼着文档同步更新。
PRESET_NOTES = {
    "清新人像 (Portra 400)": "低反差、轻微降饱和，色调近中性；肤色柔和",
    "日系空气感 (Fuji 400H)": "低饱和、通透，中调偏暖；适合小清新",
    "暖阳金调 (Kodak Gold)": "暖调，彩色预设里反差与饱和度最高",
    "冷调电影感 (Vision3 500T)": "偏暖、中等饱和（500T 是钨丝平衡胶片，面对 D65 会自然偏暖）",
    "浓郁鲜艳 (Ektar)": "色调中性偏冷，比原图略淡",
    "黑白纪实 (Tri-X)": "高反差黑白，灰轴中性",
    "自定义": "不做 LUT 变换，只响应下方手动参数",
}


def preset_tooltip():
    """把实测性格拼成预设下拉框的 tooltip，让用户在选之前就知道会得到什么。"""
    return "选择胶片风格（描述为该 LUT 的实测性格）：\n" + "\n".join(
        f"· {name}——{PRESET_NOTES[name]}" for name in PRESET_LABELS)


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
