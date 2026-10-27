# -*- coding: utf-8 -*-
"""胶片预设参数表（纯数据）。

curve 为每通道 (R, G, B) 的特性曲线参数：
  gamma    —— 中间调斜率（反差），1.0 为中性；pivot(0.18) 由曲线构造保证不动
  toe      —— 暗部软压缩强度（趾部），越大暗部越柔、黑位抬升越明显
  shoulder —— 高光软压缩强度（肩部），越大中高调越亮、亮部滚降越缓
crosstalk 为 3x3 通道串扰矩阵（按行 9 个数，行和约等于 1 以保住曝光）。
split_tone 为分离色调：shadow / highlight 为 sRGB 颜色，amt 为强度。
sat_scale 会乘到用户饱和度上（用户为 1 时即预设推荐饱和度）。
black_lift 在线性域抬升黑位。monochrome 为 True 时转黑白后再做分离色调（B&W 调色）。

参数来源：tests/calibrate_stocks.py 可微拟合标定。拟合结构为"共享曲线标量（有界）+
每通道 ±0.1 偏移"，性格主要由串扰矩阵/分离色调/饱和度承载。参考 LUT：
poptechstudio/ai-cinema-studio-engine（MIT）、YahiaAngelo/Film-Luts 的
G'MIC/Pat David 胶片集（CC BY-SA 数据，仅作标定参考不随插件分发）、
ganjmeng/dazz-retro-camera（MIT），以及用户 luts/ 收藏中的同名胶片 LUT。
每个预设条目标注了拟合所用的参考 LUT。"自定义"保持中性，未参与拟合。

本表在 characteristic_curve 改为"肩部按 x=1 归一、gamma 不再被回锚抵消"之后整体
重新标定过一次（同一套参考 LUT、同一色板、同一目标函数）；gamma 从原先的实际失效
变成了真正的中间调调节量，因此数值与旧版不可直接比较。
"""

FILM_STOCKS = {
    "清新人像 (Portra 400)": {  # 参考 tests/reference_luts/kodak_portra_400.cube
        "curve": {"gamma": (1.0284, 0.9244, 0.9443), "toe": (0.0415, 0.0413, 0.0424), "shoulder": (1.1021, 1.1065, 1.0848)},
        "crosstalk": [0.9785, 0.0706, 0.1295,
                      0.0419, 0.9815, 0.1281,
                      -0.0222, 0.2043, 0.9565],
        "split_tone": {"shadow": (0.3396, 0.0201, 0.6032), "shadow_amt": 0.0098,
                       "highlight": (0.712, 0.711, 0.7055), "highlight_amt": 0.3491},
        "sat_scale": 1.4584,
        "black_lift": (0.0018, 0.0022, 0.0018),
        "monochrome": False,
    },
    "日系空气感 (Fuji 400H)": {  # 参考 tests/reference_luts/fuji_400h.cube
        "curve": {"gamma": (0.9838, 0.838, 0.9674), "toe": (0.0092, 0.0071, 0.0169), "shoulder": (0.6064, 0.5435, 0.423)},
        "crosstalk": [0.9322, 0.1086, 0.2254,
                      0.0557, 0.9799, 0.1287,
                      0.0219, 0.1989, 0.9580],
        "split_tone": {"shadow": (0.981, 0.1474, 0.144), "shadow_amt": 0.0304,
                       "highlight": (0.9743, 0.9358, 0.8973), "highlight_amt": 0.2563},
        "sat_scale": 1.4806,
        "black_lift": (0.0026, 0.0035, 0.0028),
        "monochrome": False,
    },
    "暖阳金调 (Kodak Gold)": {  # 参考 tests/reference_luts/cpm35_kodak_gold_warm.cube
        "curve": {"gamma": (1.1672, 1.1471, 1.1445), "toe": (0.0435, 0.0405, 0.0223), "shoulder": (0.3739, 0.3194, 0.1812)},
        "crosstalk": [0.9917, -0.0922, -0.0159,
                      -0.0199, 0.9997, -0.0059,
                      -0.0026, -0.0051, 1.0000],
        "split_tone": {"shadow": (0.6143, 0.4776, 0.1796), "shadow_amt": 0.1074,
                       "highlight": (0.9617, 0.731, 0.1026), "highlight_amt": 0.1142},
        "sat_scale": 1.1062,
        "black_lift": (0.0015, 0.0004, 0.0001),
        "monochrome": False,
    },
    "冷调电影感 (Vision3 500T)": {  # 参考 kodak_vision3_500t.cube
        "curve": {"gamma": (0.982, 0.9015, 1.0769), "toe": (0.1426, 0.0022, 0.1493), "shoulder": (0.1998, 0.1963, 0.0006)},
        "crosstalk": [0.9952, 0.0630, -0.1196,
                      -0.0366, 1.0038, -0.0043,
                      -0.2612, -0.1260, 0.9098],
        "split_tone": {"shadow": (0.9577, 0.7234, 0.3606), "shadow_amt": 0.3496,
                       "highlight": (0.936, 0.7353, 0.0212), "highlight_amt": 0.1195},
        "sat_scale": 1.4979,
        "black_lift": (0.0119, 0.0032, 0.0),
        "monochrome": False,
    },
    "浓郁鲜艳 (Ektar)": {  # 参考 Neg_Kodak_Ektar_100.cube
        "curve": {"gamma": (1.309, 1.2098, 1.2168), "toe": (0.2608, 0.1775, 0.1815), "shoulder": (0.0025, 0.0033, 0.003)},
        "crosstalk": [1.0004, -0.0032, 0.0032,
                      0.0157, 0.9981, 0.0406,
                      -0.0133, 0.1930, 0.9610],
        "split_tone": {"shadow": (0.8937, 0.4815, 0.0651), "shadow_amt": 0.0115,
                       "highlight": (0.7482, 0.9745, 0.6168), "highlight_amt": 0.0475},
        "sat_scale": 1.0831,
        "black_lift": (0.0006, 0.0003, 0.0003),
        "monochrome": False,
    },
    "黑白纪实 (Tri-X)": {  # 参考 BW_Kodak_TriX_400.cube；串扰=感光权重
        "curve": {"gamma": (1.1877, 1.1836, 1.1789), "toe": (0.1088, 0.1088, 0.1085), "shoulder": (1.2856, 1.2856, 1.2858)},
        "crosstalk": [0.8096, -0.1646, 0.1987,
                      0.0328, 0.7724, 0.1700,
                      -0.2011, -0.2079, 0.7730],
        "split_tone": {"shadow": (0.8284, 0.6919, 0.944), "shadow_amt": 0.0177,
                       "highlight": (0.9648, 0.9457, 0.9809), "highlight_amt": 0.0346},
        "sat_scale": 1.0,
        "black_lift": (0.0051, 0.0053, 0.0078),
        "monochrome": True,
    },
    "自定义": {
        "curve": {"gamma": (1.02, 1.02, 1.02), "toe": (0.35, 0.35, 0.35), "shoulder": (0.40, 0.40, 0.40)},
        "crosstalk": [1.0, 0.0, 0.0,
                      0.0, 1.0, 0.0,
                      0.0, 0.0, 1.0],
        "split_tone": {"shadow": (0.0, 0.0, 0.0), "shadow_amt": 0.0,
                       "highlight": (0.0, 0.0, 0.0), "highlight_amt": 0.0},
        "sat_scale": 1.0,
        "black_lift": (0.0, 0.0, 0.0),
        "monochrome": False,
    },
}

PRESET_LABELS = list(FILM_STOCKS.keys())
