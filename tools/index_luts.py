# -*- coding: utf-8 -*-
"""给 luts/ 目录里的每个 LUT 量一个"性格"，生成 luts/README.md 索引。

为什么需要：`luts/` 里有一百多个文件，其中大半是 `Essential LUT 01`～`35`、`Cinematic LUT 01`～`17`
这类完全看不出内容的编号名。用户只能一个个试。这里用与预设同一套口径把每个 LUT 的
冷暖、饱和、反差量出来，按风格分组列表——想找"偏冷、低饱和"的直接按数值挑。

口径（与 tests/preset_character_test.py 完全一致，strength=1、不自动曝光）：
    色度   765 色板的平均 max−min，以及相对原图的倍数（>1 更浓，<1 更淡）
    暖度   全部色块的平均 R−B（>0 偏暖，<0 偏冷）
    反差   sRGB 0.30→0.70 的输出比 / 线性输入比；恒等为 0.47

用法：
    python tools/index_luts.py            # 重新生成 luts/README.md
    python tools/index_luts.py --check    # 只校验索引是否与目录一致（测试用）
"""

import argparse
import importlib.util
import math
import pathlib
import re
import sys

import torch

HERE = pathlib.Path(__file__).resolve().parent
PKG = HERE.parent
LUTS = PKG / "luts"
INDEX = LUTS / "README.md"

_spec = importlib.util.spec_from_file_location(
    "comfyui_FilmLook", PKG / "__init__.py", submodule_search_locations=[str(PKG)])
_fc_mod = sys.modules.get("comfyui_FilmLook")
if _fc_mod is None:
    _fc_mod = importlib.util.module_from_spec(_spec)
    sys.modules["comfyui_FilmLook"] = _fc_mod
    _spec.loader.exec_module(_fc_mod)
fc = _fc_mod.film_core

# 分组规则：先匹配更具体的，最后落到「其它」
GROUPS = [
    ("胶片模拟（具名）", re.compile(
        r"^(kodak|fuji|neg_|slide_|bw_|16mm|faded|instant|xpro|vintage_)", re.I)),
    ("ANALOGICA（印片/LOG 转换）", re.compile(r"^ANALOGICA", re.I)),
    ("Arthouse", re.compile(r"^Arthouse LUT", re.I)),
    ("Cinematic", re.compile(r"^Cinematic LUT", re.I)),
    ("Commercial", re.compile(r"^Commercial LUT", re.I)),
    ("Essential", re.compile(r"^Essential LUT", re.I)),
    ("Urban", re.compile(r"^Urban LUT", re.I)),
]


def group_of(name):
    for title, pat in GROUPS:
        if pat.search(name):
            return title
    return "其它"


def measure(path):
    lut = fc.parse_cube(str(path))
    lv = torch.linspace(0, 1, 9)
    r, g, b = torch.meshgrid(lv, lv, lv, indexing="ij")
    pal = torch.stack([r, g, b], dim=-1).reshape(1, -1, 1, 3)
    out = fc.apply_cube(pal, lut, 1.0)
    base = float((pal.max(-1).values - pal.min(-1).values).mean())
    chroma = float((out.max(-1).values - out.min(-1).values).mean())
    gray = torch.tensor([0.30, 0.70]).view(1, 2, 1, 1).expand(1, 2, 1, 3).contiguous()
    gout = fc.apply_cube(gray, lut, 1.0).reshape(2, 3).mean(dim=1)
    x0 = float(fc.srgb_to_linear(torch.tensor(0.30)))
    x1 = float(fc.srgb_to_linear(torch.tensor(0.70)))
    return {
        "chroma": chroma / base,
        "warm": float((out[..., 0] - out[..., 2]).mean()),
        "contrast": math.log2(float(gout[1]) / max(float(gout[0]), 1e-6)) / math.log2(x1 / x0),
    }


def collect():
    files = sorted(p for p in LUTS.iterdir() if p.suffix.lower() in (".cube", ".png"))
    rows = []
    for p in files:
        try:
            m = measure(p)
            rows.append((p.name, group_of(p.name), m, p.stat().st_size))
        except Exception as e:  # noqa: BLE001
            rows.append((p.name, group_of(p.name), None, p.stat().st_size))
            print(f"  跳过 {p.name}: {type(e).__name__}: {e}")
    return rows


def render(rows):
    order = [t for t, _ in GROUPS] + ["其它"]
    by_group = {g: [] for g in order}
    for row in rows:
        by_group.setdefault(row[1], []).append(row)

    out = ["# `luts/` 目录索引", "",
           "本文件由 `tools/index_luts.py` 自动生成，**不要手改**"
           "（`tests/luts_index_test.py` 会校验它和目录是否一致）。", "",
           "FilmLook LUT 节点会在下拉框里列出本目录的全部 `.cube` / `.png` 文件。"
           "下表把每个 LUT 的**实测性格**量了出来——编号名（`Essential LUT 07` 之类）"
           "看不出内容，直接按冷暖/饱和/反差挑更省事。", "",
           "口径与预设一致（strength=1、不自动曝光，765 色板）：", "",
           "- **色度**：相对原图的倍数，>1 更浓、<1 更淡；",
           "- **暖度**：全部色块平均 R−B，>0 偏暖、<0 偏冷；",
           "- **反差**：sRGB 0.30→0.70 的输出比 / 线性输入比，恒等为 0.47（越大越硬）。", ""]

    total = sum(len(v) for v in by_group.values())
    out.append(f"共 **{total}** 个文件。")
    out.append("")
    for g in order:
        items = by_group.get(g) or []
        if not items:
            continue
        out.append(f"## {g}（{len(items)}）")
        out.append("")
        out.append("| 文件 | 色度 | 暖度 | 反差 | 大小 |")
        out.append("|---|---|---|---|---|")
        for name, _g, m, size in sorted(items, key=lambda r: (r[2]["warm"] if r[2] else 99)):
            if m is None:
                out.append(f"| `{name}` | — | — | — | {size / 1024:.0f}KB |")
            else:
                out.append(f"| `{name}` | {m['chroma']:.2f}× | {m['warm']:+.3f} | "
                           f"{m['contrast']:.2f} | {size / 1024:.0f}KB |")
        out.append("")
    return "\n".join(out) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="只校验索引与目录一致")
    a = ap.parse_args()

    rows = collect()
    if a.check:
        if not INDEX.is_file():
            print("缺少 luts/README.md，请先运行 python tools/index_luts.py")
            return 1
        text = INDEX.read_text(encoding="utf-8")
        missing = [r[0] for r in rows if f"`{r[0]}`" not in text]
        if missing:
            print(f"索引缺少 {len(missing)} 个文件，例如：{missing[:5]}")
            return 1
        print(f"索引与目录一致（{len(rows)} 个文件）。")
        return 0

    INDEX.write_text(render(rows), encoding="utf-8")
    print(f"已写出 {INDEX}（{len(rows)} 个文件，{sum(r[3] for r in rows) / 1024 / 1024:.0f}MB）")
    warm = [r for r in rows if r[2]]
    if warm:
        hottest = max(warm, key=lambda r: r[2]["warm"])
        coolest = min(warm, key=lambda r: r[2]["warm"])
        vivid = max(warm, key=lambda r: r[2]["chroma"])
        print(f"  最暖：{hottest[0]} ({hottest[2]['warm']:+.3f})"
              f"   最冷：{coolest[0]} ({coolest[2]['warm']:+.3f})"
              f"   最浓：{vivid[0]} ({vivid[2]['chroma']:.2f}×)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
