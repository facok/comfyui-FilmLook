# -*- coding: utf-8 -*-
"""LUT 生成器的自检：朝向、单调、值域，以及"写出的 .cube 能被正确读回"。

最后一项是这里最重要的：`write_cube` 曾经把索引写成 [b,g,r]，导致红蓝轴对调。
那次错误**灰阶完全正常、色板平均色度也不变**（互换等于把输入集合重排，均值不变），
只有原色探测和"写→读往返"能抓到。本测试把这两种探针都固定下来。
"""

import importlib.util
import pathlib
import sys
import tempfile

import numpy as np
import torch

P = pathlib.Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location(
    "comfyui_FilmLook", P / "__init__.py", submodule_search_locations=[str(P)])
mod = importlib.util.module_from_spec(spec)
sys.modules["comfyui_FilmLook"] = mod
spec.loader.exec_module(mod)
fc = mod.film_core

_gen_spec = importlib.util.spec_from_file_location("gen_film_luts", P / "tools" / "gen_film_luts.py")
gen = importlib.util.module_from_spec(_gen_spec)
sys.modules["gen_film_luts"] = gen
_gen_spec.loader.exec_module(gen)

FAILED = []


def check(name, cond, detail=""):
    if cond:
        print(f"  PASS {name}{(' — ' + detail) if detail else ''}")
    else:
        FAILED.append(name)
        print(f"  FAIL {name}{(' — ' + detail) if detail else ''}")


SIZE = 9
print(f"[1] 生成 {SIZE}³ Portra 400（负片 + Kodak 2383 印片）")
neg = gen.Film(gen.load_film("kodak_portra_400", "KODAK_PORTRA_400"))
prt = gen.Film(gen.load_film("kodak_2383", "KODAK_2383"))
lut = gen.build_lut(neg, prt, size=SIZE)
check("形状正确", lut.shape == (SIZE, SIZE, SIZE, 3), str(lut.shape))
check("值域在 [0,1]", float(lut.min()) >= 0 and float(lut.max()) <= 1,
      f"[{lut.min():.3f},{lut.max():.3f}]")
check("无 NaN/Inf", bool(np.isfinite(lut).all()))

print("[2] 朝向：纯原色必须得到对应色（红蓝互换是隐蔽错误）")
for idx, tag in ((0, "红"), (1, "绿"), (2, "蓝")):
    v = lut[(SIZE - 1) * (idx == 0), (SIZE - 1) * (idx == 1), (SIZE - 1) * (idx == 2)]
    check(f"纯{tag}输出以{tag}为主", int(np.argmax(v)) == idx,
          f"-> ({v[0]:.3f},{v[1]:.3f},{v[2]:.3f})")

print("[3] 灰阶单调")
g = np.array([lut[i, i, i] for i in range(SIZE)])
mono = all((g[i + 1] >= g[i]).all() for i in range(SIZE - 1))
check("灰阶逐级不递减", mono)
check("黑位足够低", float(g[0].mean()) < 0.15, f"{g[0].mean():.3f}")
check("白位足够高", float(g[-1].mean()) > 0.75, f"{g[-1].mean():.3f}")

print("[4] 写→读往返一致（能抓住 write_cube 的轴序错误）")
with tempfile.TemporaryDirectory() as td:
    out = pathlib.Path(td) / "t.cube"
    gen.write_cube(out, lut, "roundtrip")
    parsed = fc.parse_cube(str(out))
    check("LUT_3D_SIZE 正确", int(parsed["size"]) == SIZE, str(parsed["size"]))
    step = SIZE - 1
    grid = torch.tensor([[i / step, j / step, k / step]
                         for i in range(SIZE) for j in range(SIZE) for k in range(SIZE)]).view(-1, 1, 3)
    got = fc.apply_cube(grid.view(1, -1, 1, 3), parsed, 1.0).reshape(SIZE, SIZE, SIZE, 3)
    # apply_cube 的输入按 [h,w,c]→[r,g,b] 处理，这里逐点比对
    worst = float(torch.abs(got - torch.from_numpy(lut).float()).max())
    check("读回后的采样与原始数组一致", worst < 2e-4, f"max|Δ|={worst:.2e}")

print("\n" + ("生成器自检全部通过。" if not FAILED else f"失败 {len(FAILED)} 项：{FAILED}"))
sys.exit(1 if FAILED else 0)
