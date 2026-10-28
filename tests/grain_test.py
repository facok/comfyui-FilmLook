# -*- coding: utf-8 -*-
"""颗粒与「强度 0 = 原图」回归测试。

锁定的行为：
  1. seed 真的改变颗粒图案，且同一 seed 可复现；
  2. 同一批次内每帧颗粒互相独立（不是整块平移的同一张图案）；
  3. 彩色颗粒模式下 R/G/B 三通道图案互相独立；
  4. strength=0 时 grain / 一键节点 / 黑白预设都严格等于原图。

注意第 2、3 条不能只用「输出不同」来判断：silvergrain 的 cell 种子是
(y*65536 + x + offset)，offset 差 1 只是整块图案平移一个 cell，肉眼与统计几乎相同。
所以这里用皮尔逊相关系数，而不是 max|Δ|。
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

FAILED = []


def check(name, cond, detail=""):
    if cond:
        print(f"  PASS {name}{(' — ' + detail) if detail else ''}")
    else:
        FAILED.append(name)
        print(f"  FAIL {name}{(' — ' + detail) if detail else ''}")


try:
    import silvergrain  # noqa: F401
    HAS_SG = True
except ImportError:
    HAS_SG = False

if not HAS_SG:
    print("未安装 silvergrain，跳过颗粒相关断言（仅验证强度 0 的恒等性）。")

from comfyui_FilmLook.grain_nodes import apply_grain, _mix_seed  # noqa: E402

FLAT = torch.full((1, 256, 256, 3), 0.5)
BATCH = torch.full((3, 256, 256, 3), 0.5)


def grain(x, seed=0, gtype="亮度颗粒 (保色彩)", strength=1.0, sharpen=0.0):
    return apply_grain(x, "标准 (ISO 400)", gtype, strength, "均衡", seed, sharpen)


def corr(a, b):
    a = a - a.mean()
    b = b - b.mean()
    return float((a * b).mean() / (a.std() * b.std() + 1e-12))


if HAS_SG:
    print("[1] seed 生效且可复现")
    a = grain(FLAT, seed=0)
    b = grain(FLAT, seed=123456)
    check("不同 seed 图案不同", float((a - b).abs().max()) > 0.02,
          f"max|Δ|={float((a - b).abs().max()):.4f}")
    check("同 seed 可复现", torch.equal(a, grain(FLAT, seed=0)))

    print("[2] 批次逐帧独立（相关系数必须接近 0，而不是平移）")
    frames = grain(BATCH, seed=0)
    c01 = corr(frames[0], frames[1])
    c02 = corr(frames[0], frames[2])
    c12 = corr(frames[1], frames[2])
    check("帧0 vs 帧1 去相关", abs(c01) < 0.35, f"corr={c01:+.3f}")
    check("帧0 vs 帧2 去相关", abs(c02) < 0.35, f"corr={c02:+.3f}")
    check("帧1 vs 帧2 去相关", abs(c12) < 0.35, f"corr={c12:+.3f}")

    print("[3] 彩色颗粒：三通道图案互相独立")
    rgb_out = grain(FLAT, gtype="彩色颗粒 (胶片感更强)")
    c_rg = corr(rgb_out[..., 0], rgb_out[..., 1])
    c_rb = corr(rgb_out[..., 0], rgb_out[..., 2])
    check("R vs G 去相关", abs(c_rg) < 0.35, f"corr={c_rg:+.3f}")
    check("R vs B 去相关", abs(c_rb) < 0.35, f"corr={c_rb:+.3f}")

    print("[4] seed 派生函数本身可靠（相邻帧号必须雪崩）")
    seeds = [_mix_seed(0, i) for i in range(4)]
    distinct = len(set(seeds)) == 4
    spread = min(abs(a - b) for a in seeds for b in seeds if a != b)
    check("相邻帧号派生出互异且相距很远的种子", distinct and spread > 1 << 20,
          f"seeds={[hex(s) for s in seeds]} 最小间距={spread:#x}")

print("[5] strength=0 严格等于原图")
img = torch.rand(1, 64, 64, 3, generator=torch.Generator().manual_seed(9))
if HAS_SG:
    check("grain strength=0（含默认 sharpen=0.3）",
          torch.equal(apply_grain(img, "标准 (ISO 400)", "亮度颗粒 (保色彩)", 0.0, "均衡", 0, 0.3), img))
    check("grain strength=0 且彩色模式",
          torch.equal(apply_grain(img, "标准 (ISO 400)", "彩色颗粒 (胶片感更强)", 0.0, "均衡", 0), img))
check("film color strength=0（黑白预设）",
      torch.equal(mod.film_core.apply_film_color(img, mod.film_stocks.preset_entry("黑白纪实 (Tri-X)"),
                                                 strength=0.0), img))
check("film color strength=0（彩色预设）",
      torch.equal(mod.film_core.apply_film_color(img, mod.film_stocks.preset_entry("清新人像 (Portra 400)"),
                                                 strength=0.0), img))
check("光晕 strength=0", torch.equal(mod.film_core.apply_halation(img, strength=0.0), img))
check("镜头节点全 0（柔光+暗角）",
      torch.equal(mod.FilmLookLens().execute(img, 0.0, 0.0)[0], img))
check("LUT 节点 strength=0",
      torch.equal(mod.FilmLookLUT().execute(img, "kodak_vision3_500t.cube", 0.0, "")[0], img))

if HAS_SG:
    print("[6] strength 单调：颗粒幅度随强度递增")
    stds = [float(grain(FLAT, strength=s).std()) for s in (0.25, 0.5, 1.0)]
    check("颗粒幅度单调", stds[0] < stds[1] < stds[2], " → ".join(f"{v:.4f}" for v in stds))

    print("[7] 一键节点全 0 严格等于原图")
    one = mod.FilmLookOneClick()
    out = one.execute(img, "自定义", "标准 (ISO 400)", 0.0, 0.0, False, 0.0, 1.0, 0.0, 0.0, 0)[0]
    check("OneClick 全 0 恒等", torch.equal(out, img),
          f"max|Δ|={float((out - img).abs().max()):.2e}")

print("\n" + ("全部颗粒/恒等测试通过。" if not FAILED else f"失败 {len(FAILED)} 项：{FAILED}"))
sys.exit(1 if FAILED else 0)
