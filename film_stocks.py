# -*- coding: utf-8 -*-
"""胶片预设参数表（纯数据）。

curve 为每通道 (R, G, B) 的特性曲线参数：
  gamma    —— 中间调斜率（反差），1.0 为中性
  toe      —— 暗部软压缩强度（趾部），越大暗部越柔
  shoulder —— 高光软压缩强度（肩部），越大高光滚降越明显
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
"""

FILM_STOCKS = {
    "清新人像 (Portra 400)": {  # 拟合参考 kodak_portra_400.cube（真 Portra 400，G'MIC/Pat David 集）
        "curve": {"gamma": (0.6654, 0.6648, 0.6656), "toe": (0.4560, 0.6296, 0.6152), "shoulder": (0.7137, 0.6017, 0.5760)},
        "crosstalk": [0.9779, 0.0743, 0.1270,
                      0.0387, 0.9810, 0.1308,
                      -0.0264, 0.1831, 0.9651],
        "split_tone": {"shadow": (0.3510, 0.0121, 0.6008), "shadow_amt": 0.0121,
                       "highlight": (0.7122, 0.7097, 0.7073), "highlight_amt": 0.3489},
        "sat_scale": 1.3812,
        "black_lift": (0.0064, 0.0075, 0.0056),
        "monochrome": False,
    },
    "日系空气感 (Fuji 400H)": {  # 拟合参考 fuji_400h.cube（真 400H，G'MIC/Pat David 集）
        "curve": {"gamma": (0.6641, 0.6665, 0.6664), "toe": (0.0649, 0.1585, 0.0642), "shoulder": (0.3227, 0.2047, 0.2362)},
        "crosstalk": [0.9354, 0.0936, 0.2270,
                      0.0544, 0.9789, 0.1328,
                      0.0196, 0.1879, 0.9628],
        "split_tone": {"shadow": (0.9760, 0.1344, 0.2423), "shadow_amt": 0.0334,
                       "highlight": (0.9732, 0.9370, 0.8969), "highlight_amt": 0.3033},
        "sat_scale": 1.4475,
        "black_lift": (0.0067, 0.0095, 0.0069),
        "monochrome": False,
    },
    "暖阳金调 (Kodak Gold)": {  # 拟合参考 cpm35_kodak_gold_warm.cube（真 Gold 200 暖调，dazz-retro-camera）
        "curve": {"gamma": (0.9839, 0.7999, 0.7991), "toe": (0.2480, 0.2287, 0.0611), "shoulder": (0.3957, 0.3709, 0.2044)},
        "crosstalk": [0.9849, -0.1230, -0.0137,
                      -0.0254, 0.9993, -0.0077,
                      -0.0039, -0.0061, 1.0000],
        "split_tone": {"shadow": (0.6318, 0.4888, 0.1397), "shadow_amt": 0.1259,
                       "highlight": (0.9545, 0.7407, 0.2003), "highlight_amt": 0.1290},
        "sat_scale": 1.0358,
        "black_lift": (0.0009, 0.0, 0.0001),
        "monochrome": False,
    },
    "冷调电影感 (Vision3 500T)": {  # 拟合参考 kodak_vision3_500t.cube
        "curve": {"gamma": (1.2956, 1.1115, 1.1158), "toe": (0.3058, 0.1985, 0.1100), "shoulder": (0.2088, 0.1403, 0.0136)},
        "crosstalk": [1.0024, 0.0414, -0.0498,
                      -0.0171, 1.0004, -0.0125,
                      -0.2409, 0.0587, 0.8974],
        "split_tone": {"shadow": (0.9657, 0.7453, 0.0199), "shadow_amt": 0.3447,
                       "highlight": (0.9206, 0.7723, 0.0413), "highlight_amt": 0.1404},
        "sat_scale": 1.4516,
        "black_lift": (0.0180, 0.0032, 0.0001),
        "monochrome": False,
    },
    "浓郁鲜艳 (Ektar)": {  # 拟合参考 Neg_Kodak_Ektar_100.cube（真 Ektar 负片 LUT）
        "curve": {"gamma": (1.0761, 0.8994, 0.9027), "toe": (0.0720, 0.0614, 0.0628), "shoulder": (0.0042, 0.0010, 0.0011)},
        "crosstalk": [0.9998, -0.0057, 0.0046,
                      0.0125, 0.9983, 0.0384,
                      -0.0145, 0.1897, 0.9623],
        "split_tone": {"shadow": (0.7929, 0.7812, 0.0606), "shadow_amt": 0.0161,
                       "highlight": (0.7146, 0.9794, 0.5555), "highlight_amt": 0.0545},
        "sat_scale": 1.0763,
        "black_lift": (0.0012, 0.0004, 0.0006),
        "monochrome": False,
    },
    "黑白纪实 (Tri-X)": {  # 拟合参考 BW_Kodak_TriX_400.cube（真 Tri-X LUT；串扰=感光权重）
        "curve": {"gamma": (1.2328, 1.2327, 1.2326), "toe": (0.8356, 0.8356, 0.8356), "shoulder": (0.8479, 0.8479, 0.8479)},
        "crosstalk": [0.8100, -0.1636, 0.1932,
                      0.0377, 0.7734, 0.1646,
                      -0.2046, -0.2112, 0.7661],
        "split_tone": {"shadow": (0.8161, 0.6261, 0.9601), "shadow_amt": 0.0234,
                       "highlight": (0.9657, 0.9382, 0.9893), "highlight_amt": 0.0535},
        "sat_scale": 1.0,
        "black_lift": (0.0075, 0.0078, 0.0111),
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
