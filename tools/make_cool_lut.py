# -*- coding: utf-8 -*-
"""合成「冷调电影感」LUT：以 MIT 授权的 vision3_500t 为底，去掉钨丝偏色后做冷调分级。

为什么不用库里现成的冷调 LUT：`luts/` 里确实有几十条偏冷的（Cinematic LUT 03/05/10 等），
但它们属于用户自备收藏，来源与许可不明，不能随插件分发。这里改用**派生**：基底是 MIT 的
vision3_500t（真实胶片响应），先把它在灰轴上的钨丝偏色中和回中性，再叠加冷调分级——
既保留胶片的色调层次，又得到干净的「青蓝暗部 + 中性高光」。

链路：
    500T LUT 查表 → 灰轴中和(逐通道增益) → 线性域：暗部偏青、整体降饱和 → 输出
其中暗部偏青用亮度权重 smoothstep 控制，只影响暗部，不污染肤色（肤色在中高亮度）。

用法：python tools/make_cool_lut.py [--preview]
"""
import argparse
import importlib.util
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
PKG = HERE.parent
BASE = PKG / "luts" / "fuji_eterna_250d.cube"
OUT = PKG / "presets" / "cool_cinema.cube"

_spec = importlib.util.spec_from_file_location(
    "comfyui_FilmLook", PKG / "__init__.py", submodule_search_locations=[str(PKG)])
mod = sys.modules.get("comfyui_FilmLook")
if mod is None:
    mod = importlib.util.module_from_spec(_spec)
    sys.modules["comfyui_FilmLook"] = mod
    _spec.loader.exec_module(mod)
fc = mod.film_core

LUMA = (0.2126, 0.7152, 0.0722)


def srgb_to_linear(x):
    return torch_where(x)


def torch_where(x):
    import torch
    x = torch.as_tensor(x, dtype=torch.float32)
    return torch.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(x):
    import torch
    return torch.where(x <= 0.0031308, x * 12.92, 1.055 * x.clamp_min(1e-8) ** (1 / 2.4) - 0.055)


def build(size=33, teal=1.2, cool=0.2, sat=1.0, contrast=1.12):
    import torch
    base = fc.parse_cube(str(BASE))

    # 1) 用 500T 自己的灰轴求中和增益：让 sRGB 0.5 的中性灰输出回到中性
    g = torch.linspace(0.2, 0.8, 13).view(1, -1, 1, 1).expand(1, -1, 1, 3).contiguous()
    gray_out = fc.apply_cube(g, base, 1.0)[0, :, 0, :]          # (13,3)
    target = g[0, :, 0, 0]                                       # 输入灰阶（sRGB）
    # 逐通道增益：让输出灰阶尽量贴近输入灰阶（最小二乘斜率）
    nei = target - target.mean()
    gains = []
    for c in range(3):
        oc = gray_out[:, c] - gray_out[:, c].mean()
        k = float((nei * oc).sum() / (oc * oc).sum().clamp_min(1e-8))
        gains.append(1.0 / max(k, 1e-3))
    gains = torch.tensor(gains).view(1, 1, 1, 3)
    # 归一化：中灰亮度不变
    w = torch.tensor(LUMA).view(1, 1, 1, 3)
    gains = gains / (gains * w).sum() * 1.0

    # 2) 生成网格并查表
    lin = torch.linspace(0, 1, size)
    r, gg, b = torch.meshgrid(lin, lin, lin, indexing="ij")
    rgb = torch.stack([r, gg, b], axis=-1).reshape(-1, 1, 3).unsqueeze(0)   # (1,N,1,3)
    out = fc.apply_cube(rgb, base, 1.0)
    out = (out * gains).clamp(0, 1)

    x = srgb_to_linear(out)
    lum = (x * w).sum(dim=-1, keepdim=True)

    # 3) 冷调分级：必须用**乘性**增益。加性偏移会把 0 抬起来，黑位变成青灰、整幅糊成一片。
    #    乘性增益让黑仍然是黑，只有暗部中间调偏青，高光保持中性。
    shadow_w = (1.0 - (lum / 0.35).clamp(0, 1)) ** 2          # 只在暗部起作用
    tt = float(teal) * shadow_w
    x = x * torch.cat([1.0 - 0.38 * tt, 1.0 - 0.02 * tt, 1.0 + 0.48 * tt], dim=-1)

    cc = float(cool) * (1.0 - lum)                            # 整体冷偏，高光尽量少动
    x = x * torch.cat([1.0 - 0.55 * cc, torch.ones_like(cc), 1.0 + 0.55 * cc], dim=-1)

    # 4) 柔和肩部，避免高光硬切
    knee = 0.82
    over = (x - knee).clamp_min(0)
    x = torch.where(x > knee, knee + (1 - knee) * (1 - torch.exp(-over / (1 - knee))), x)

    res = linear_to_srgb(x.clamp_min(0)).clamp(0, 1).reshape(size, size, size, 3)
    return res


def measure(lut, size):
    import torch
    lv = torch.linspace(0, 1, 9)
    r, g, b = torch.meshgrid(lv, lv, lv, indexing="ij")
    pal = torch.stack([r, g, b], dim=-1).reshape(1, -1, 1, 3)
    out = fc.apply_cube(pal, {"type": "3d", "table": lut, "size": size,
                              "domain_min": torch.zeros(3), "domain_max": torch.ones(3)}, 1.0)
    base = float((pal.max(-1).values - pal.min(-1).values).mean())
    chroma = float((out.max(-1).values - out.min(-1).values).mean())
    gray = torch.tensor([0.30, 0.70]).view(1, 2, 1, 1).expand(1, 2, 1, 3).contiguous()
    gout = fc.apply_cube(gray, {"type": "3d", "table": lut, "size": size,
                                "domain_min": torch.zeros(3), "domain_max": torch.ones(3)}, 1.0
                         ).reshape(2, 3).mean(dim=1)
    x0 = float(fc.srgb_to_linear(torch.tensor(0.30)))
    x1 = float(fc.srgb_to_linear(torch.tensor(0.70)))
    import math
    return {"ratio": chroma / base,
            "warm": float((out[..., 0] - out[..., 2]).mean()),
            "contrast": math.log2(float(gout[1]) / max(float(gout[0]), 1e-6)) / math.log2(x1 / x0)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=33)
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--teal", type=float, default=1.2)
    ap.add_argument("--cool", type=float, default=0.2)
    ap.add_argument("--sat", type=float, default=1.0)
    ap.add_argument("--contrast", type=float, default=1.12)
    a = ap.parse_args()

    lut = build(a.size, teal=a.teal, cool=a.cool, sat=a.sat, contrast=a.contrast)
    m = measure(lut, a.size)
    print(f"合成结果：色度 {m['ratio']:.2f}×  暖度 {m['warm']:+.3f}  反差 {m['contrast']:.2f}")
    if a.write:
        table = {"table": lut, "size": a.size, "domain_min": np.zeros(3), "domain_max": np.ones(3)}
        _write(OUT, lut, "FilmLook Cool Cinema")
        print(f"已写出 {OUT}")
    return 0


def _write(path, lut, title):
    size = lut.shape[0]
    lines = [f'TITLE "{title}"', f"LUT_3D_SIZE {size}", "DOMAIN_MIN 0.0 0.0 0.0",
             "DOMAIN_MAX 1.0 1.0 1.0", ""]
    for b in range(size):
        for g in range(size):
            for r in range(size):
                c = lut[r, g, b]
                lines.append(f"{float(c[0]):.6f} {float(c[1]):.6f} {float(c[2]):.6f}")
    pathlib.Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
