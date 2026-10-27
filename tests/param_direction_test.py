# -*- coding: utf-8 -*-
"""方向回归测试：锁定 temperature / tint 的符号、互反性、亮度守恒与可用幅度。

存在的理由：原先 tint 的符号是反的（正值给出偏绿），而既有测试只检查值域与 NaN，
方向错了也发现不了。改动 film_core.apply_temp_tint 时请保持本文件通过。
"""

import importlib.util
import pathlib
import sys

import torch

PKG_DIR = pathlib.Path(__file__).resolve().parent.parent


def load_plugin():
    spec = importlib.util.spec_from_file_location(
        "comfyui_FilmLook", PKG_DIR / "__init__.py", submodule_search_locations=[str(PKG_DIR)])
    mod = importlib.util.module_from_spec(spec)
    sys.modules["comfyui_FilmLook"] = mod
    spec.loader.exec_module(mod)
    return mod


mod = load_plugin()
film_core = mod.film_core

SRGB_GRAY = torch.full((1, 4, 4, 3), 0.5)
LIN_GRAY = film_core.srgb_to_linear(SRGB_GRAY)
FAILED = []


def check(name, cond, detail=""):
    if cond:
        print(f"  PASS {name}{(' — ' + detail) if detail else ''}")
    else:
        FAILED.append(name)
        print(f"  FAIL {name}{(' — ' + detail) if detail else ''}")


def temp(srgb_in, t, n):
    """对 sRGB 输入做 temp/tint，返回 sRGB 输出（管线内部是线性域）。"""
    lin = film_core.srgb_to_linear(srgb_in)
    return film_core.linear_to_srgb(film_core.apply_temp_tint(lin, t, n)).clamp(0.0, 1.0)


def rgb_of(o):
    return [float(v) for v in o.reshape(-1, 3).mean(dim=0)]


print("[1] temperature 方向：+ 暖（R>G>B），− 冷（B>G>R）")
for t in (0.5, 1.0):
    r, g, b = rgb_of(temp(SRGB_GRAY, t, 0.0))
    check(f"temperature=+{t} 偏暖", r > g > b, f"R={r:.4f} G={g:.4f} B={b:.4f}")
for t in (-0.5, -1.0):
    r, g, b = rgb_of(temp(SRGB_GRAY, t, 0.0))
    check(f"temperature={t} 偏冷", b > g > r, f"R={r:.4f} G={g:.4f} B={b:.4f}")

print("[2] tint 方向：+ 洋红（G 最低、R≈B），− 绿（G 最高、R≈B）")
for n in (0.5, 1.0):
    r, g, b = rgb_of(temp(SRGB_GRAY, 0.0, n))
    check(f"tint=+{n} 偏洋红", g < r - 1e-3 and g < b - 1e-3 and abs(r - b) < 1e-5,
          f"R={r:.4f} G={g:.4f} B={b:.4f}")
for n in (-0.5, -1.0):
    r, g, b = rgb_of(temp(SRGB_GRAY, 0.0, n))
    check(f"tint={n} 偏绿", g > r + 1e-3 and g > b + 1e-3 and abs(r - b) < 1e-5,
          f"R={r:.4f} G={g:.4f} B={b:.4f}")

print("[3] 单调性")
vals = [-1.0, -0.6, -0.3, 0.0, 0.3, 0.6, 1.0]
rbs = []
for t in vals:
    r, _, b = rgb_of(temp(SRGB_GRAY, t, 0.0))
    rbs.append(r - b)
check("温度轴 R-B 随 temperature 递增", all(a < b for a, b in zip(rbs, rbs[1:])),
      " → ".join(f"{v:+.3f}" for v in rbs))
gavg = []
for n in vals:
    r, g, b = rgb_of(temp(SRGB_GRAY, 0.0, n))
    gavg.append(g - (r + b) / 2)
check("色调轴 G-avg 随 tint 递减", all(a > b for a, b in zip(gavg, gavg[1:])),
      " → ".join(f"{v:+.3f}" for v in gavg))

print("[4] 正负行程互反：+x 后接 -x 只应留下均匀缩放，色比必须完全还原")
x = torch.rand(1, 8, 8, 3, generator=torch.Generator().manual_seed(4)) * 0.8 + 0.1
for t in (0.3, 0.6, 1.0):
    back = film_core.apply_temp_tint(film_core.apply_temp_tint(x, t, 0.0), -t, 0.0)
    scale = back / x
    spread = float((scale.amax() - scale.amin()) / scale.mean())
    check(f"temperature ±{t} 往返色比还原", spread < 1e-5,
          f"通道缩放离散 {spread:.1e}，亮度偏移 {float(scale.mean()) - 1:+.2%}")
for n in (0.3, 1.0):
    back = film_core.apply_temp_tint(film_core.apply_temp_tint(x, 0.0, n), 0.0, -n)
    scale = back / x
    spread = float((scale.amax() - scale.amin()) / scale.mean())
    check(f"tint ±{n} 往返色比还原", spread < 1e-5,
          f"通道缩放离散 {spread:.1e}，亮度偏移 {float(scale.mean()) - 1:+.2%}")

print("[5] 亮度守恒与零位恒等")
for t, n in ((1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0), (0.6, -0.6)):
    o = film_core.apply_temp_tint(LIN_GRAY, t, n)
    dlum = abs(float(film_core.luminance(o).mean() - film_core.luminance(LIN_GRAY).mean()))
    check(f"temp={t}, tint={n} 中灰亮度不变", dlum < 1e-6, f"Δluma={dlum:.1e}")
check("零位严格恒等", torch.equal(film_core.apply_temp_tint(LIN_GRAY, 0.0, 0.0), LIN_GRAY))

print("[6] 可用幅度（sRGB 0.5 中灰，满行程必须肉眼可见且不失稳）")
r, _, b = rgb_of(temp(SRGB_GRAY, 1.0, 0.0))
check("温度满行程", 0.10 <= r - b <= 0.30, f"R-B={r - b:+.3f}")
r, g, b = rgb_of(temp(SRGB_GRAY, 0.0, 1.0))
check("色调满行程", 0.05 <= (r + b) / 2 - g <= 0.25, f"G-(R+B)/2={g - (r + b) / 2:+.3f}")

print("[7] 节点端到端（FilmLookColor / 自定义 / strength=1 / auto_exposure=off）")
node = mod.FilmLookColor()


def run_node(t, n):
    return node.execute(SRGB_GRAY, "自定义", 1.0, False, 0.0, 1.0, 1.0, t, n, 0.0, 1.0)[0]


b0 = rgb_of(run_node(0.0, 0.0))
r, _, b = rgb_of(run_node(1.0, 0.0))
check("节点 temperature=+1 更暖", (r - b) > (b0[0] - b0[2]) + 0.05,
      f"R-B: {b0[0] - b0[2]:+.4f} → {r - b:+.4f}")
r, _, b = rgb_of(run_node(-1.0, 0.0))
check("节点 temperature=-1 更冷", (r - b) < (b0[0] - b0[2]) - 0.05,
      f"R-B: {b0[0] - b0[2]:+.4f} → {r - b:+.4f}")
r, g, b = rgb_of(run_node(0.0, 1.0))
check("节点 tint=+1 偏洋红", g < r - 0.01 and g < b - 0.01, f"R={r:.4f} G={g:.4f} B={b:.4f}")
r, g, b = rgb_of(run_node(0.0, -1.0))
check("节点 tint=-1 偏绿", g > r + 0.01 and g > b + 0.01, f"R={r:.4f} G={g:.4f} B={b:.4f}")

print("\n" + ("全部方向测试通过。" if not FAILED else f"失败 {len(FAILED)} 项：{FAILED}"))
sys.exit(1 if FAILED else 0)
