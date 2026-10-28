# 生成一张接近实拍场景的测试图，跑 FilmLookOneClick 各预设并输出 PNG 供目检
import importlib.util
import os
import pathlib
import sys

import torch
from PIL import Image

PKG_DIR = pathlib.Path(__file__).resolve().parent.parent
OUT = pathlib.Path(__file__).resolve().parent / "output"
OUT.mkdir(exist_ok=True)

spec = importlib.util.spec_from_file_location(
    "comfyui_FilmLook", PKG_DIR / "__init__.py", submodule_search_locations=[str(PKG_DIR)])
mod = importlib.util.module_from_spec(spec)
sys.modules["comfyui_FilmLook"] = mod
spec.loader.exec_module(mod)


def make_scene(w=768, h=512):
    """夜景街拍感 + 标定目检区：天空渐变 + 路灯高光 + 暗部地面 + 肤色块 +
    大面积中性灰墙面(0.2-0.3) + 饱和红绿蓝色块 + 底部灰阶条。"""
    ys = torch.linspace(0, 1, h).view(h, 1).expand(h, w)
    xs = torch.linspace(0, 1, w).view(1, w).expand(h, w)
    img = torch.zeros(h, w, 3)
    img[..., 0] = 0.05 + 0.10 * (1 - ys) + 0.05 * xs   # 天空暗蓝
    img[..., 1] = 0.06 + 0.12 * (1 - ys)
    img[..., 2] = 0.10 + 0.22 * (1 - ys)
    img[ys > 0.7] *= 0.35                               # 地面暗部
    skin = (xs > 0.55) & (xs < 0.8) & (ys > 0.25) & (ys < 0.65)
    img[skin] = torch.tensor([0.72, 0.55, 0.44])        # 肤色亮块
    lamp = ((xs - 0.22) ** 2 + (ys - 0.38) ** 2) < 0.012
    img[lamp] = 1.0                                     # 灯泡
    lamp2 = ((xs - 0.9) ** 2 + (ys - 0.3) ** 2) < 0.004
    img[lamp2] = 1.0
    wall = (xs > 0.06) & (xs < 0.40) & (ys > 0.52) & (ys < 0.80)
    img[wall] = (0.22 + 0.06 * (1 - ys))[wall].unsqueeze(-1)  # 中性灰墙面 0.22-0.28
    for i, color in enumerate([(0.75, 0.08, 0.08), (0.10, 0.65, 0.12), (0.10, 0.20, 0.85)]):
        block = (xs > 0.66 + i * 0.10) & (xs < 0.74 + i * 0.10) & (ys > 0.68) & (ys < 0.82)
        img[block] = torch.tensor(color)                # 饱和色块 红/绿/蓝
    ramp = ys > 0.90                                    # 底部灰阶条（16 级）
    steps = (xs * 16).long().clamp(0, 15) / 15.0
    img[ramp] = steps[ramp].unsqueeze(-1)
    return img.unsqueeze(0).clamp(0, 1)


def save(tensor, name):
    arr = (tensor[0].cpu().numpy() * 255).clip(0, 255).astype("uint8")
    Image.fromarray(arr).save(OUT / name)
    print("saved", name)


scene = make_scene()
save(scene, "00_original.png")

node = mod.FilmLookOneClick()
presets = ["清新人像 (Portra 400)", "日系空气感 (Fuji 400H)", "暖阳金调 (Kodak Gold)",
           "夜景暖调 (Vision3 500T)", "自然淡彩 (Ektar 100)", "黑白纪实 (Tri-X)"]
for i, p in enumerate(presets):
    out = node.execute(scene, p, "标准 (ISO 400)", 0.85, 0.4, True, 0.3, 1.0, 0.2, 0.2, 42)[0]
    save(out, f"{i+1:02d}_{p.split(' ')[0]}.png")
print("DONE")
