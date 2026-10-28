# 质量链路综合审计：性能、显存、极端输入、非常规尺寸
import importlib.util
import pathlib
import sys
import time

import torch

PKG_DIR = pathlib.Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location(
    "comfyui_FilmLook", PKG_DIR / "__init__.py", submodule_search_locations=[str(PKG_DIR)])
mod = importlib.util.module_from_spec(spec)
sys.modules["comfyui_FilmLook"] = mod
spec.loader.exec_module(mod)

from comfyui_FilmLook import film_core
from comfyui_FilmLook.film_stocks import preset_entry
from comfyui_FilmLook.grain_nodes import apply_grain

stock = preset_entry("夜景暖调 (Vision3 500T)")


def section(t):
    print(f"\n=== {t} ===")


# ---------- 1. 性能与显存 ----------
section("性能/显存（调色+光晕为 GPU torch；颗粒为 silvergrain）")
for h, w in [(512, 512), (1024, 1024), (2048, 2048), (2160, 3840)]:
    img = torch.rand(1, h, w, 3, device="cuda")
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = film_core.apply_film_color(img, stock)
    out = film_core.apply_halation(out)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    peak = torch.cuda.max_memory_allocated() / 2**20
    print(f"  {w}x{h}: 调色+光晕 {t1-t0:.2f}s, 峰值显存 {peak:.0f}MB")

img = torch.rand(1, 1024, 1024, 3, device="cuda")
t0 = time.perf_counter()
apply_grain(img, "标准 (ISO 400)", "亮度颗粒 (保色彩)", 1.0, "均衡", 0, 0.3)
print(f"  1024x1024 颗粒: {time.perf_counter()-t0:.2f}s")

# ---------- 2. 极端输入数值稳定性 ----------
section("极端输入")
cases = {
    "全黑": torch.zeros(1, 64, 64, 3),
    "全白": torch.ones(1, 64, 64, 3),
    "超界HDR(0~2)": torch.rand(1, 64, 64, 3) * 2,
    "负值(-0.5~0.5)": torch.rand(1, 64, 64, 3) - 0.5,
    "含NaN": torch.rand(1, 64, 64, 3).masked_fill(torch.rand(1, 64, 64, 3) > 0.999, float("nan")),
    "1x1像素": torch.rand(1, 1, 1, 3),
    "极长条(16x4096)": torch.rand(1, 16, 4096, 3),
    "RGBA": torch.rand(1, 64, 64, 4),
    "大批次x8": torch.rand(8, 256, 256, 3),
}
for name, t in cases.items():
    try:
        o1 = film_core.apply_film_color(t, stock)
        o2 = film_core.apply_halation(t)
        bad = torch.isnan(o1).any() or torch.isnan(o2).any()
        print(f"  {name}: 调色 range=[{float(o1.min()):.3f},{float(o1.max()):.3f}] "
              f"光晕 range=[{float(o2.min()):.3f},{float(o2.max()):.3f}] {'NaN!' if bad else 'OK'}")
    except Exception as e:
        print(f"  {name}: 异常 {type(e).__name__}: {e}")

# ---------- 3. 幂等性/确定性 ----------
section("确定性（同输入同参数两次运行）")
img = torch.rand(1, 256, 256, 3, generator=torch.Generator().manual_seed(7))
a = film_core.apply_film_color(img, stock)
b = film_core.apply_film_color(img, stock)
print(f"  调色两次输出一致: {torch.equal(a, b)}")
a = film_core.apply_halation(img, strength=0.5)
b = film_core.apply_halation(img, strength=0.5)
print(f"  光晕两次输出一致: {torch.equal(a, b)}")

# ---------- 4. 颗粒对纯色图的均值保持（不应系统性提亮/压暗）----------
section("颗粒统计特性")
flat = torch.full((1, 512, 512, 3), 0.3)
out = apply_grain(flat, "标准 (ISO 400)", "亮度颗粒 (保色彩)", 1.0, "速度优先", 0, 0.0)
print(f"  0.3 平坦图颗粒后均值: {float(out.mean()):.4f}（不应偏离太多），标准差 {float(out.std()):.4f}")
flat2 = torch.full((1, 512, 512, 3), 0.8)
out2 = apply_grain(flat2, "标准 (ISO 400)", "亮度颗粒 (保色彩)", 1.0, "速度优先", 0, 0.0)
print(f"  0.8 平坦图颗粒后均值: {float(out2.mean()):.4f}，标准差 {float(out2.std()):.4f}")

print("\n审计完成")
