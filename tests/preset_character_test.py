# -*- coding: utf-8 -*-
"""预设描述必须与实测一致。

`film_stocks.PRESET_NOTES` 里的描述会被拼进节点 tooltip，用户在选之前就靠它判断。
本测试重新测量每个预设的色度 / 暖度 / 反差，逐条核对描述里的说法——
换 LUT、改管线、改 `apply_film_color` 都会让这里失败，从而逼着文档同步。

指标（都在 strength=1、关闭自动曝光下测）：
    ratio    765 色板平均色度 / 原图平均色度（>1 比原图浓，<1 更淡）
    warm     全部色块的平均 R−B（>0 偏暖，<0 偏冷）
    contrast sRGB 0.30→0.70 的输出比 / 线性输入比；恒等管线取值为 0.47
"""

import importlib.util
import math
import pathlib
import sys

import torch

P = pathlib.Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location(
    "comfyui_FilmLook", P / "__init__.py", submodule_search_locations=[str(P)])
mod = importlib.util.module_from_spec(spec)
sys.modules["comfyui_FilmLook"] = mod
spec.loader.exec_module(mod)
fc = mod.film_core
stocks = mod.film_stocks

FAILED = []


def check(name, cond, detail=""):
    if cond:
        print(f"  PASS {name}{(' — ' + detail) if detail else ''}")
    else:
        FAILED.append(name)
        print(f"  FAIL {name}{(' — ' + detail) if detail else ''}")


def measure(label):
    st = stocks.preset_entry(label)
    lv = torch.linspace(0, 1, 9)
    r, g, b = torch.meshgrid(lv, lv, lv, indexing="ij")
    pal = torch.stack([r, g, b], dim=-1).reshape(1, -1, 1, 3)
    out = fc.apply_film_color(pal, st, strength=1.0, auto_exposure=False)
    chroma = float((out.max(-1).values - out.min(-1).values).mean())
    base = float((pal.max(-1).values - pal.min(-1).values).mean())
    gray = torch.tensor([0.30, 0.70]).view(1, 2, 1, 1).expand(1, 2, 1, 3).contiguous()
    gout = fc.apply_film_color(gray, st, strength=1.0, auto_exposure=False).reshape(2, 3).mean(dim=1)
    x0 = float(fc.srgb_to_linear(torch.tensor(0.30)))
    x1 = float(fc.srgb_to_linear(torch.tensor(0.70)))
    return {
        "chroma": chroma,
        "ratio": chroma / base,
        "warm": float((out[..., 0] - out[..., 2]).mean()),
        "contrast": math.log2(float(gout[1]) / max(float(gout[0]), 1e-6)) / math.log2(x1 / x0),
        "black": float(fc.apply_film_color(torch.zeros(1, 1, 1, 3), st,
                                           strength=1.0, auto_exposure=False).mean()),
    }


M = {label: measure(label) for label in stocks.PRESET_LABELS}

print("[1] 每个预设都必须有实测描述")
check("PRESET_NOTES 覆盖全部预设且无多余",
      set(stocks.PRESET_NOTES) == set(stocks.PRESET_LABELS),
      f"缺 {set(stocks.PRESET_LABELS) - set(stocks.PRESET_NOTES)}")
tip = stocks.preset_tooltip()
check("tooltip 里逐个列出预设", all(n in tip for n in stocks.PRESET_LABELS))

print("[2] 描述与实测逐条对齐")
check("清新人像「低反差」", M["清新人像 (Portra 400)"]["contrast"] < 0.46,
      f"contrast={M['清新人像 (Portra 400)']['contrast']:.2f}")
check("清新人像「轻微降饱和」", 0.72 <= M["清新人像 (Portra 400)"]["ratio"] <= 0.92,
      f"ratio={M['清新人像 (Portra 400)']['ratio']:.2f}")
check("清新人像「色调近中性」", abs(M["清新人像 (Portra 400)"]["warm"]) < 0.05,
      f"warm={M['清新人像 (Portra 400)']['warm']:+.3f}")

check("日系空气感「低饱和」", M["日系空气感 (Fuji 400H)"]["ratio"] < 0.85,
      f"ratio={M['日系空气感 (Fuji 400H)']['ratio']:.2f}")
check("日系空气感「中调偏暖」", M["日系空气感 (Fuji 400H)"]["warm"] > 0.04,
      f"warm={M['日系空气感 (Fuji 400H)']['warm']:+.3f}")

check("暖阳金调「暖调」", M["暖阳金调 (Kodak Gold)"]["warm"] > 0.02,
      f"warm={M['暖阳金调 (Kodak Gold)']['warm']:+.3f}")
check("暖阳金调「彩色预设里反差最高」",
      M["暖阳金调 (Kodak Gold)"]["contrast"] >= max(
          M[k]["contrast"] for k in M if k not in ("暖阳金调 (Kodak Gold)", "黑白纪实 (Tri-X)", "自定义")),
      f"contrast={M['暖阳金调 (Kodak Gold)']['contrast']:.2f}")

check("夜景暖调「偏暖」", M["夜景暖调 (Vision3 500T)"]["warm"] > 0.10,
      f"warm={M['夜景暖调 (Vision3 500T)']['warm']:+.3f}")
check("夜景暖调「中等饱和」", 0.80 <= M["夜景暖调 (Vision3 500T)"]["ratio"] <= 1.05,
      f"ratio={M['夜景暖调 (Vision3 500T)']['ratio']:.2f}")

check("自然淡彩「比原图略淡」", M["自然淡彩 (Ektar 100)"]["ratio"] < 0.90,
      f"ratio={M['自然淡彩 (Ektar 100)']['ratio']:.2f}")
check("自然淡彩「色调中性偏冷」", M["自然淡彩 (Ektar 100)"]["warm"] < 0.05,
      f"warm={M['自然淡彩 (Ektar 100)']['warm']:+.3f}")

check("黑白纪实「黑白」", M["黑白纪实 (Tri-X)"]["chroma"] < 1e-3,
      f"chroma={M['黑白纪实 (Tri-X)']['chroma']:.4f}")

print("[3] 全表排序断言（防止换错 LUT）")
check("Vision3 500T 是全场最暖",
      M["夜景暖调 (Vision3 500T)"]["warm"] >= max(v["warm"] for v in M.values()))
check("Fuji 400H 是彩色预设里最淡",
      M["日系空气感 (Fuji 400H)"]["ratio"] <= min(
          v["ratio"] for k, v in M.items() if k not in ("黑白纪实 (Tri-X)",)))
check("自定义是恒等管线",
      abs(M["自定义"]["ratio"] - 1.0) < 0.01 and abs(M["自定义"]["warm"]) < 1e-3
      and abs(M["自定义"]["contrast"] - 0.47) < 0.02,
      f"ratio={M['自定义']['ratio']:.2f} warm={M['自定义']['warm']:+.3f} "
      f"contrast={M['自定义']['contrast']:.2f}")

print("\n实测汇总：")
print(f"  {'预设':24s} {'色度':>6s} {'相对':>5s} {'暖度':>7s} {'反差':>5s} {'黑位':>6s}")
for k, v in M.items():
    print(f"  {k:24s} {v['chroma']:6.3f} {v['ratio']:5.2f} {v['warm']:+7.3f} "
          f"{v['contrast']:5.2f} {v['black']:6.3f}")

print("\n[4] 旧预设名仍然可用（改名不得破坏已保存的工作流）")
_grade = mod.FilmLookColor()
_kw = dict(strength=1.0, auto_exposure=True, brightness=0.0, contrast=1.0, saturation=1.0,
           temperature=0.0, tint=0.0, preserve_luminance=0.0, style_boost=1.0)
_probe = torch.rand(1, 8, 8, 3, generator=torch.Generator().manual_seed(21))
for old, new in stocks.LEGACY_PRESETS.items():
    check(f"「{old}」解析为「{new}」", stocks.resolve_preset(old) == new)
    a = _grade.execute(_probe, old, **_kw)[0]
    b = _grade.execute(_probe, new, **_kw)[0]
    check("  旧名与新名输出逐像素一致", torch.equal(a, b))
    check("  旧名通过 VALIDATE_INPUTS",
          mod.FilmLookColor.VALIDATE_INPUTS(preset=old) is True)
    check("  OneClick 同样放行旧名",
          mod.FilmLookOneClick.VALIDATE_INPUTS(preset=old) is True)
_unknown = mod.FilmLookColor.VALIDATE_INPUTS(preset="根本没有这个预设")
check("未知名字被拒绝且提示可选值", isinstance(_unknown, str) and "可选" in _unknown)

print("\n" + ("全部预设描述校验通过。" if not FAILED else f"失败 {len(FAILED)} 项：{FAILED}"))
sys.exit(1 if FAILED else 0)
