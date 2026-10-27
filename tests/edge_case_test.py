# -*- coding: utf-8 -*-
"""边界与几何回归测试。

覆盖四类曾经真实崩溃 / 失真的情形：
  1. 细长图与超小图（4096x16、512x1、1x1）在光晕 / 柔光 / 颗粒锐化里触发
     F.pad(mode="reflect") 的 "Padding size should be less than ..." 报错；
  2. 单帧超过 2^24 像素时 torch.quantile 的 "input tensor is too large"；
  3. 颗粒节点压暗画面，且压暗量随分辨率变化（grain_sigma 被写死为绝对像素值）；
  4. 暗角在单像素轴上坐标退化成 -1；光晕 / 柔光按 height 缩放导致竖图强于横图。
"""

import importlib.util
import pathlib
import sys
import tempfile

import torch

PKG_DIR = pathlib.Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location(
    "comfyui_FilmLook", PKG_DIR / "__init__.py", submodule_search_locations=[str(PKG_DIR)])
mod = importlib.util.module_from_spec(spec)
sys.modules["comfyui_FilmLook"] = mod
spec.loader.exec_module(mod)
fc = mod.film_core
STOCK = mod.film_stocks.FILM_STOCKS["清新人像 (Portra 400)"]
FAILED = []


def check(name, cond, detail=""):
    if cond:
        print(f"  PASS {name}{(' — ' + detail) if detail else ''}")
    else:
        FAILED.append(name)
        print(f"  FAIL {name}{(' — ' + detail) if detail else ''}")


print("[1] 细长 / 超小图的形状矩阵（曾抛 reflect padding 错误）")
cube = pathlib.Path(tempfile.mkdtemp()) / "id.cube"
cube.write_text("LUT_3D_SIZE 2\n0 0 0\n1 0 0\n0 1 0\n1 1 0\n0 0 1\n1 0 1\n0 1 1\n1 1 1\n", encoding="ascii")
nodes = [
    ("Color", lambda t: mod.FilmLookColor().execute(t, "清新人像 (Portra 400)", 0.85, True, 0, 1, 1, 0, 0, 0, 1)[0]),
    ("Halation", lambda t: mod.FilmLookHalation().execute(t, 0.5, "夸张", True, 0.85)[0]),
    ("Lens", lambda t: mod.FilmLookLens().execute(t, 0.5, 0.5)[0]),
    ("LUT", lambda t: mod.FilmLookLUT().execute(t, "自定义路径", 1.0, str(cube))[0]),
]
try:
    import silvergrain  # noqa: F401
    nodes.append(("Grain", lambda t: mod.FilmLookGrain().execute(
        t, "标准 (ISO 400)", "亮度颗粒 (保色彩)", 1.0, "速度优先", 0.3, 0)[0]))
except ImportError:
    pass

for name, fn in nodes:
    bad = []
    for h, w in ((1, 1), (2, 1), (1, 512), (512, 1), (3, 3), (16, 4096), (4096, 16)):
        try:
            o = fn(torch.rand(1, h, w, 3))
            if torch.isnan(o).any():
                bad.append(f"{h}x{w}:NaN")
        except Exception as e:  # noqa: BLE001
            bad.append(f"{h}x{w}:{type(e).__name__}")
    check(f"{name} 通过 7 种极端形状", not bad, ", ".join(bad))

print("[2] 超过 2^24 像素的单帧（曾抛 quantile 上限错误）")
big = torch.full((1, 17_000_000, 1, 3), 0.214)
gain = fc.auto_exposure_gain(big)
check("1700 万像素 auto_exposure_gain 不报错", abs(float(gain) - 0.917) < 0.01, f"gain={float(gain):.4f}")
samples = fc._frame_samples(torch.rand(1, 20_000_000, 1, 1))
check("_frame_samples 抽样封顶", samples.shape[1] <= 1_000_000, f"shape={tuple(samples.shape)}")

print("[3] 颗粒不得改变整体亮度，且不随分辨率漂移")
try:
    from comfyui_FilmLook.grain_nodes import apply_grain
    shifts = []
    for size in (128, 256, 512):
        out = apply_grain(torch.full((1, size, size, 3), 0.5), "标准 (ISO 400)",
                          "亮度颗粒 (保色彩)", 1.0, "速度优先", 0, 0.0)
        shifts.append(float(out.mean()) - 0.5)
    check("各分辨率压暗量均 <1%", max(abs(s) for s in shifts) < 0.01,
          " / ".join(f"{s:+.4f}" for s in shifts))
    check("各分辨率之间偏差 <0.3%", max(shifts) - min(shifts) < 0.003,
          f"极差={max(shifts) - min(shifts):.4f}")
except ImportError:
    print("  跳过（未安装 silvergrain）")

print("[4] 暗角几何")
one = fc.apply_vignette(torch.full((1, 512, 1, 3), 0.5), strength=0.8)
check("单像素轴不被当成角落", float(one.mean()) > 0.48, f"均值={float(one.mean()):.4f}")
land = fc.apply_vignette(torch.full((1, 64, 256, 3), 0.5), strength=0.8)
port = fc.apply_vignette(torch.full((1, 256, 64, 3), 0.5), strength=0.8)
check("横竖暗角一致", abs(float(land.mean()) - float(port.mean())) < 1e-5,
      f"{float(land.mean()):.4f} vs {float(port.mean()):.4f}")
check("中心不受影响", abs(float(land[0, 32, 128, 0]) - 0.5) < 1e-6)

print("[5] 光晕几何：竖图与横图强度一致")
a = fc.apply_halation(torch.zeros(1, 256, 1024, 3).index_put_((torch.tensor([0]), torch.tensor([128]), torch.tensor([512])), torch.tensor(1.0)),
                      strength=1.0, scales=(0.02,), auto_threshold=False, threshold=0.5)
b = fc.apply_halation(torch.zeros(1, 1024, 256, 3).index_put_((torch.tensor([0]), torch.tensor([512]), torch.tensor([128])), torch.tensor(1.0)),
                      strength=1.0, scales=(0.02,), auto_threshold=False, threshold=0.5)
check("横竖光晕能量一致", abs(float(a.sum()) - float(b.sum())) < 1e-3,
      f"{float(a.sum()):.2f} vs {float(b.sum()):.2f}")

print("[6] 滑块拉到边界的组合扫描（不应出现 NaN 或越界）")
import itertools  # noqa: E402
node = mod.FilmLookColor()
img = torch.rand(1, 32, 32, 3, generator=torch.Generator().manual_seed(1))
bad = []
combos = 0
for preset in ("清新人像 (Portra 400)", "黑白纪实 (Tri-X)", "自定义"):
    for br, ct, sat, tm, tn in itertools.product((-2.0, 2.0), (0.2, 2.0), (0.0, 2.0),
                                                 (-1.0, 1.0), (-1.0, 1.0)):
        combos += 1
        try:
            out = node.execute(img, preset, 1.0, True, br, ct, sat, tm, tn, 1.0, 2.0)[0]
            if torch.isnan(out).any() or float(out.min()) < 0.0 or float(out.max()) > 1.0:
                bad.append((preset, br, ct, sat, tm, tn))
        except Exception as e:  # noqa: BLE001
            bad.append((preset, type(e).__name__, str(e)[:40]))
check(f"{combos} 组极端组合无 NaN / 越界", not bad, str(bad[:3]))

print("[7] LUT 解析缓存：命中要快、文件变了要失效")
import shutil  # noqa: E402
import time  # noqa: E402

src = PKG_DIR / "luts" / "kodak_vision3_500t.cube"
if src.is_file():
    tmp_cube = pathlib.Path(tempfile.mkdtemp()) / "cache_probe.cube"
    shutil.copy(src, tmp_cube)
    t0 = time.perf_counter()
    first = fc.parse_cube(str(tmp_cube))
    t_first = time.perf_counter() - t0
    t0 = time.perf_counter()
    second = fc.parse_cube(str(tmp_cube))
    t_second = time.perf_counter() - t0
    check("重复解析返回同一对象", first is second, f"{t_first * 1000:.2f}ms → {t_second * 1000:.3f}ms")
    check("缓存命中显著更快", t_second < max(t_first * 0.2, 1e-4),
          f"{t_first * 1000:.2f}ms → {t_second * 1000:.3f}ms")
    time.sleep(0.02)
    tmp_cube.write_text("LUT_3D_SIZE 2\n0 0 0\n1 0 0\n0 1 0\n1 1 0\n0 0 1\n1 0 1\n0 1 1\n1 1 1\n", encoding="ascii")
    check("文件改动后缓存失效", fc.parse_cube(str(tmp_cube))["size"] == 2,
          f"size {first['size']} → 2")

print("\n" + ("全部边界测试通过。" if not FAILED else f"失败 {len(FAILED)} 项：{FAILED}"))
sys.exit(1 if FAILED else 0)
