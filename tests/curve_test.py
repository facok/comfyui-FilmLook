# -*- coding: utf-8 -*-
"""特性曲线回归测试：锚点、单调性、以及 gamma 是否真的控制中间调斜率。

存在的理由：旧实现用"端点归一化 + 幂函数回锚 pivot"，回锚指数恰好等于 1/gamma，
把 gamma 精确抵消——纯曲线时 gamma 0.75~1.35 与恒等曲线的差只有 1e-7，真实预设里
也只剩约 1/4 的调节力。下面第 3、4 组断言就是为这条 bug 设的防回归线。
"""

import importlib.util
import pathlib
import sys

import torch

PKG_DIR = pathlib.Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location(
    "comfyui_FilmLook", PKG_DIR / "__init__.py", submodule_search_locations=[str(PKG_DIR)])
mod = importlib.util.module_from_spec(spec)
sys.modules["comfyui_FilmLook"] = mod
spec.loader.exec_module(mod)
fc = mod.film_core

FAILED = []


def check(name, cond, detail=""):
    if cond:
        print(f"  PASS {name}{(' — ' + detail) if detail else ''}")
    else:
        FAILED.append(name)
        print(f"  FAIL {name}{(' — ' + detail) if detail else ''}")


def curve_at(srgb_levels, gamma, toe, shoulder):
    x = torch.tensor(srgb_levels, dtype=torch.float32).view(1, -1, 1, 1).expand(1, -1, 1, 3).contiguous()
    return fc.characteristic_curve(fc.srgb_to_linear(x), (toe,) * 3, (shoulder,) * 3, (gamma,) * 3)[0, :, 0, 0]


def log_slope(levels, out):
    l0, l1 = fc.srgb_to_linear(torch.tensor([levels[0]])).item(), fc.srgb_to_linear(torch.tensor([levels[1]])).item()
    return float(torch.log2(out[1] / out[0]) / torch.log2(torch.tensor(l1 / l0)))


print("[1] 锚点：黑 0、白 1、中灰 0.18")
y = curve_at([0.0, 0.5, 1.0], 1.2, 0.5, 0.4)
check("f(0) = 0", float(y[0]) < 1e-6, f"{float(y[0]):.2e}")
check("f(1) = 1", abs(float(y[2]) - 1.0) < 1e-4, f"{float(y[2]):.6f}")
mid = float(fc.characteristic_curve(torch.full((1, 1, 1, 3), 0.18), (0.5,) * 3, (0.4,) * 3, (1.2,) * 3)[0, 0, 0, 0])
check("f(0.18) ≈ 0.18", abs(mid - 0.18) < 0.01, f"{mid:.4f}")

print("[2] 单调性与值域")
lv = [i / 63 for i in range(64)]
for g, t, s in ((1.0, 0.0, 0.0), (1.3, 0.8, 1.2), (0.75, 1.2, 0.2)):
    o = curve_at(lv, g, t, s)
    mono = all(float(a) < float(b) for a, b in zip(o, o[1:]))
    check(f"g={g}, toe={t}, shoulder={s} 单调", mono,
          f"范围 [{float(o.min()):.3f}, {float(o.max()):.3f}]")
    check(f"g={g}, toe={t}, shoulder={s} 无 NaN/越界",
          not torch.isnan(o).any() and float(o.min()) >= 0.0 and float(o.max()) <= 1.0)

print("[3] gamma 必须改变中间调斜率（旧实现此处为 0）")
for label, t, s, pair in (("有肩部（跨 pivot）", 0.5, 0.5, (0.12, 0.30)),
                          ("纯曲线（暗部段）", 0.0, 0.0, (0.03, 0.10))):
    s_lo = log_slope(pair, curve_at(list(pair), 0.8, t, s))
    s_hi = log_slope(pair, curve_at(list(pair), 1.25, t, s))
    check(f"{label}：gamma 0.8→1.25 使斜率上升", s_hi > s_lo * 1.10,
          f"斜率 {s_lo:.3f} → {s_hi:.3f}")

print("[4] 肩部只作用于 pivot 以上，不牵动暗部")
low = [0.0, 0.12, 0.25, 0.35]   # 均低于 pivot(线性 0.18 ≈ sRGB 0.46)
a = curve_at(low, 1.0, 0.3, 0.2)
b = curve_at(low, 1.0, 0.3, 1.2)
check("暗部与肩部无关", float((a - b).abs().max()) < 1e-6, f"max|Δ|={float((a - b).abs().max()):.1e}")
hi_a = curve_at([0.6, 0.8], 1.0, 0.3, 0.2)
hi_b = curve_at([0.6, 0.8], 1.0, 0.3, 1.2)
check("亮部随肩部抬升", all(float(x) < float(y) for x, y in zip(hi_a, hi_b)),
      f"{[round(float(v), 3) for v in hi_a]} → {[round(float(v), 3) for v in hi_b]}")

print("[5] 各预设的中灰不漂移（自动曝光关闭）")
gray = torch.full((1, 1, 1, 3), 0.5)
for label, stock in mod.film_stocks.FILM_STOCKS.items():
    out = fc.apply_film_color(gray, stock, strength=1.0, auto_exposure=False)
    lum = float(fc.luminance(out))
    check(f"{label} 中灰 0.5 → {lum:.4f}", abs(lum - 0.5) < 0.12, "")

print("\n" + ("全部曲线测试通过。" if not FAILED else f"失败 {len(FAILED)} 项：{FAILED}"))
sys.exit(1 if FAILED else 0)
