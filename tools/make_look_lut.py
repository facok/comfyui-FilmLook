# -*- coding: utf-8 -*-
"""从零合成一条「冷调电影感」LUT：不依赖任何现成 LUT，全部由算法定义。

链路（每一步都有物理/观感依据，参数可调）：

    sRGB 编码值
      → 解码到线性
      → 胶片式色调曲线（对数域 S 曲线：趾部软压缩 + 肩部滚降 + 中间调 gamma）
      → 冷调分级（**乘性**增益，保证黑还是黑）：
            暗部偏青蓝、整体色温下压、高光轻微回暖保护肤色
      → 饱和度与自然饱和度
      → 编码回 sRGB
      → 采样成 33³ .cube

为什么不用加性偏移：加性会把 0 抬起来，黑位变成青灰、整幅糊成一片；乘性增益让 0 保持 0。

用法：
    python tools/make_look_lut.py                     # 用默认参数生成 presets/cool_cinema.cube
    python tools/make_look_lut.py --teal 2.4 --cool 0.4   # 调浓淡
"""

import argparse
import math
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
PKG = HERE.parent
OUT = PKG / "presets" / "cool_cinema.cube"

LUMA = np.array([0.2126, 0.7152, 0.0722])


def srgb_to_linear(x):
    x = np.clip(x, 0.0, None)
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(x):
    x = np.clip(x, 0.0, None)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * x ** (1 / 2.4) - 0.055)


def filmic(x, gamma=1.15, toe=0.55, shoulder=0.32, pivot=0.18):
    """对数域胶片曲线：gamma 定中间调斜率，toe/shoulder 是两端软压缩系数。

    f(pivot)=pivot 由构造保证；f(0)=0、f(1)≈1 由趾部/肩部归一化保证。
    """
    p = math.log2(pivot)
    d = gamma * (np.log2(np.clip(x, 1e-6, None)) - p)

    def comp(k, v):
        k = max(k, 1e-6)
        return np.log1p(k * v) / k

    hi = comp(shoulder, np.clip(d, 0.0, None)) * ((-p) / comp(shoulder, -gamma * p))
    lo = comp(toe, np.clip(-d, 0.0, None))
    y = 2.0 ** (p + hi - lo)
    y0 = 2.0 ** (p - comp(toe, -gamma * (math.log2(1e-6) - p)))
    return np.clip((y - y0) / (1.0 - y0), 0.0, None)


def grade(lin, teal=1.8, cool=0.30, warm_hi=0.35, sat=0.96):
    """冷调分级。全部用乘性增益。

    注意所有中间权重都保持 (N,) 形状，最后 `np.stack(..., axis=-1)` 成 (N,3) 再与 lin 相乘。
    一旦让权重带出 (N,1) 维度，stack 会得到 (N,1,3)，和 (N,3) 广播成 (N,N,3)——
    数组瞬间涨到千万级，表现为"脚本卡死"。
    """
    lum = (lin * LUMA).sum(axis=-1)                                  # (N,)

    # 整体色温下压：越暗影响越大，纯白不动
    c = cool * (1.0 - np.clip(lum, 0, 1))
    lin = lin * np.stack([1.0 - 0.55 * c, np.ones_like(c), 1.0 + 0.60 * c], axis=-1)

    # 暗部偏青：只在暗部（lum<0.35）起作用
    s = (1.0 - np.clip(lum / 0.35, 0, 1)) ** 2
    t = teal * 0.30 * s
    lin = lin * np.stack([1.0 - 1.05 * t, 1.0 - 0.05 * t, 1.0 + 1.30 * t], axis=-1)

    # 高光轻微回暖，护住肤色与灯泡（青橙对比的另一半）
    h = np.clip((lum - 0.45) / 0.55, 0, 1) ** 2
    w = warm_hi * 0.12 * h
    lin = lin * np.stack([1.0 + 1.10 * w, 1.0 + 0.15 * w, 1.0 - 0.75 * w], axis=-1)

    # 自然饱和度：已经很饱和的地方几乎不动
    lum2 = (lin * LUMA).sum(axis=-1)[..., None]
    spread = lin.max(axis=-1)[..., None] - lin.min(axis=-1)[..., None]
    factor = 1.0 - (1.0 - sat) * (1.0 - np.clip(spread, 0, 1))
    lin = lum2 + (lin - lum2) * factor
    return np.clip(lin, 0.0, None)


def shoulder_roll(x, knee=0.85):
    """高光柔和收口，避免 R 通道先削顶。"""
    over = np.clip(x - knee, 0.0, None)
    return np.where(x > knee, knee + (1.0 - knee) * (1.0 - np.exp(-over / (1.0 - knee))), x)


def build(size=33, **kw):
    grid = np.linspace(0.0, 1.0, size)
    r, g, b = np.meshgrid(grid, grid, grid, indexing="ij")     # [r, g, b] 序
    rgb = np.stack([r, g, b], axis=-1).reshape(-1, 3)

    lin = filmic(srgb_to_linear(rgb), **{k: v for k, v in kw.items()
                                         if k in ("gamma", "toe", "shoulder")})
    lin = grade(lin, **{k: v for k, v in kw.items()
                        if k in ("teal", "cool", "warm_hi", "sat")})
    lin = shoulder_roll(lin)
    out = np.clip(linear_to_srgb(lin), 0.0, 1.0)
    return out.reshape(size, size, size, 3)


def measure(lut):
    size = lut.shape[0]
    grid = np.linspace(0, 1, 9)
    r, g, b = np.meshgrid(grid, grid, grid, indexing="ij")
    pal = np.stack([r, g, b], axis=-1).reshape(-1, 3)
    out = sample(lut, pal)
    base = float((pal.max(-1) - pal.min(-1)).mean())
    chroma = float((out.max(-1) - out.min(-1)).mean())

    step = size - 1
    g0 = sample(lut, np.array([[0.30, 0.30, 0.30]]))[0]
    g1 = sample(lut, np.array([[0.70, 0.70, 0.70]]))[0]
    x0 = float(srgb_to_linear(np.array(0.30)))
    x1 = float(srgb_to_linear(np.array(0.70)))
    return {"ratio": chroma / base,
            "warm": float((out[:, 0] - out[:, 2]).mean()),
            "contrast": math.log2(float(g1.mean()) / max(float(g0.mean()), 1e-6))
            / math.log2(x1 / x0),
            "black": float(sample(lut, np.zeros((1, 3)))[0].mean()),
            "white": float(sample(lut, np.ones((1, 3)))[0].mean())}


def sample(lut, rgb):
    """三线性采样，rgb 为 (N,3)；lut 是 [r,g,b] 序，与 .cube 的存储序无关。"""
    n = lut.shape[0] - 1
    p = np.clip(rgb, 0, 1) * n
    i0 = np.floor(p).astype(int)
    i1 = np.minimum(i0 + 1, n)
    f = p - i0                                  # (N,3)
    c = np.zeros((len(rgb), 3))
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                w = ((f if dx else 1 - f) * (f if dy else 1 - f) * (f if dz else 1 - f)).prod(axis=1, keepdims=True)
                ix = np.stack([i1[:, 0] if dx else i0[:, 0],
                               i1[:, 1] if dy else i0[:, 1],
                               i1[:, 2] if dz else i0[:, 2]], axis=1)
                c += w * lut[ix[:, 0], ix[:, 1], ix[:, 2]]
    return c


def write_cube(path, lut, title):
    """写 .cube：数据序必须是**红变化最快**（外层 b、中层 g、内层 r）。"""
    size = lut.shape[0]
    lines = [f'TITLE "{title}"', f"LUT_3D_SIZE {size}",
             "DOMAIN_MIN 0.0 0.0 0.0", "DOMAIN_MAX 1.0 1.0 1.0", ""]
    for bi in range(size):
        for gi in range(size):
            for ri in range(size):
                c = lut[ri, gi, bi]
                lines.append(f"{c[0]:.6f} {c[1]:.6f} {c[2]:.6f}")
    pathlib.Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=33)
    ap.add_argument("--teal", type=float, default=1.8)
    ap.add_argument("--cool", type=float, default=0.30)
    ap.add_argument("--warm-hi", type=float, default=0.35)
    ap.add_argument("--sat", type=float, default=0.96)
    ap.add_argument("--gamma", type=float, default=1.15)
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()

    lut = build(a.size, teal=a.teal, cool=a.cool, warm_hi=a.warm_hi, sat=a.sat, gamma=a.gamma)
    m = measure(lut)
    print(f"色度 {m['ratio']:.2f}×  暖度 {m['warm']:+.3f}  反差 {m['contrast']:.2f}  "
          f"黑位 {m['black']:.3f}  白位 {m['white']:.3f}")

    # 朝向自检：纯原色必须得到对应色（写文件前就能抓出红蓝对调）
    for idx, tag in ((0, "红"), (1, "绿"), (2, "蓝")):
        probe = np.zeros((1, 3))
        probe[0, idx] = 1.0
        c = sample(lut, probe)[0]
        assert int(np.argmax(c)) == idx, f"纯{tag}输出异常：{c}"

    if a.write:
        write_cube(OUT, lut, "FilmLook Cool Cinema")
        print(f"已写出 {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
