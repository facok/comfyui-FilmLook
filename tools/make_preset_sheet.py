# -*- coding: utf-8 -*-
"""生成 docs/presets.png：同一张合成场景过一遍所有预设的对照图。

用合成场景而不是照片，是为了可复现、无版权问题，而且场景里刻意放了**胶片最敏感的四样东西**：
中性灰墙面（看黑位/色偏）、肤色块（看肤色是否发灰）、路灯高光（看高光滚降）、
底部 16 级灰阶（看反差与暗部层次）。

- 与 tests/demo_render.py 的区别：那个是逐张出 PNG 供本地目检，这个是拼成一张对比图给文档用。
- 用到中文字体（Windows 雅黑/黑体）；找不到时自动退回英文标签，不会崩。

用法：python tools/make_preset_sheet.py
"""

import importlib.util
import pathlib
import sys

import torch
from PIL import Image, ImageDraw, ImageFont

HERE = pathlib.Path(__file__).resolve().parent
PKG = HERE.parent
OUT = PKG / "docs" / "presets.png"

_spec = importlib.util.spec_from_file_location(
    "comfyui_FilmLook", PKG / "__init__.py", submodule_search_locations=[str(PKG)])
mod = sys.modules.get("comfyui_FilmLook")
if mod is None:
    mod = importlib.util.module_from_spec(_spec)
    sys.modules["comfyui_FilmLook"] = mod
    _spec.loader.exec_module(mod)

PANEL_W, PANEL_H = 384, 256
LABEL_H = 30
COLS = 4

FONT_CANDIDATES = [
    ("C:/Windows/Fonts/msyh.ttc", "微软雅黑"),
    ("C:/Windows/Fonts/simhei.ttf", "黑体"),
    ("/System/Library/Fonts/PingFang.ttc", "苹方"),
    ("/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc", "Noto CJK"),
]


def load_font(size):
    for path, _name in FONT_CANDIDATES:
        if pathlib.Path(path).is_file():
            try:
                return ImageFont.truetype(path, size), True
            except Exception:  # noqa: BLE001
                continue
    return ImageFont.load_default(), False


def make_scene(w=PANEL_W, h=PANEL_H):
    """合成场景：夜空渐变 + 两盏路灯 + 肤色块 + 中性灰墙 + 饱和色块 + 底部灰阶。"""
    ys = torch.linspace(0, 1, h).view(h, 1).expand(h, w)
    xs = torch.linspace(0, 1, w).view(1, w).expand(h, w)
    img = torch.zeros(h, w, 3)
    img[..., 0] = 0.05 + 0.10 * (1 - ys) + 0.05 * xs
    img[..., 1] = 0.06 + 0.12 * (1 - ys)
    img[..., 2] = 0.10 + 0.22 * (1 - ys)
    img[ys > 0.7] *= 0.35
    skin = (xs > 0.55) & (xs < 0.8) & (ys > 0.25) & (ys < 0.65)
    img[skin] = torch.tensor([0.72, 0.55, 0.44])
    lamp = ((xs - 0.22) ** 2 + (ys - 0.38) ** 2) < 0.012
    img[lamp] = 1.0
    lamp2 = ((xs - 0.9) ** 2 + (ys - 0.3) ** 2) < 0.004
    img[lamp2] = 1.0
    wall = (xs > 0.06) & (xs < 0.40) & (ys > 0.52) & (ys < 0.80)
    img[wall] = (0.22 + 0.06 * (1 - ys))[wall].unsqueeze(-1)
    for i, color in enumerate([(0.75, 0.08, 0.08), (0.10, 0.65, 0.12), (0.10, 0.20, 0.85)]):
        block = (xs > 0.66 + i * 0.10) & (xs < 0.74 + i * 0.10) & (ys > 0.68) & (ys < 0.82)
        img[block] = torch.tensor(color)
    ramp = ys > 0.90
    steps = (xs * 16).long().clamp(0, 15) / 15.0
    img[ramp] = steps[ramp].unsqueeze(-1)
    return img.unsqueeze(0).clamp(0, 1)


def to_image(t):
    arr = (t[0].cpu().numpy() * 255).clip(0, 255).astype("uint8")
    return Image.fromarray(arr)


def main():
    font, has_cjk = load_font(15)
    node = mod.FilmLookColor()
    scene = make_scene()

    # 与 README 的预设表一致：只放"开箱即用"的六个风格，参数全默认（strength=1 才看得出差别）
    panels = [("原图", scene)]
    for name in mod.film_stocks.PRESET_LABELS:
        if name == "自定义":
            continue
        out = node.execute(scene, name, 1.0, True, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0)[0]
        panels.append((name if has_cjk else name.split(" ")[0].encode("ascii", "ignore").decode() or "preset",
                       out))

    rows = (len(panels) + COLS - 1) // COLS
    sheet = Image.new("RGB", (COLS * PANEL_W, rows * (PANEL_H + LABEL_H)), (18, 18, 20))
    draw = ImageDraw.Draw(sheet)
    for i, (label, tensor) in enumerate(panels):
        cx = (i % COLS) * PANEL_W
        cy = (i // COLS) * (PANEL_H + LABEL_H)
        sheet.paste(to_image(tensor), (cx, cy))
        text = label if has_cjk else label
        try:
            tw = draw.textlength(text, font=font)
        except Exception:  # noqa: BLE001
            tw = len(text) * 8
        draw.text((cx + (PANEL_W - tw) / 2, cy + PANEL_H + 6), text, fill=(235, 235, 235), font=font)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(OUT)
    print(f"已写出 {OUT}（{sheet.width}x{sheet.height}，{OUT.stat().st_size / 1024:.0f}KB，"
          f"字体 {'中文' if has_cjk else 'ASCII 回退'}）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
