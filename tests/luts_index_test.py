# -*- coding: utf-8 -*-
"""`luts/README.md` 必须与目录内容、实测数值完全一致。

索引是自动生成的，一旦有人加了/删了/替换了 LUT 却没重新生成，用户就会按过期数据挑文件。
本测试直接重算一遍并逐字比对——生成是确定性的（同一份 LUT、同一套口径、CPU 采样），
所以可以用严格相等。
"""

import importlib.util
import pathlib
import sys

P = pathlib.Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("index_luts", P / "tools" / "index_luts.py")
tool = importlib.util.module_from_spec(spec)
sys.modules["index_luts"] = tool
spec.loader.exec_module(tool)

FAILED = []


def check(name, cond, detail=""):
    if cond:
        print(f"  PASS {name}{(' — ' + detail) if detail else ''}")
    else:
        FAILED.append(name)
        print(f"  FAIL {name}{(' — ' + detail) if detail else ''}")


rows = tool.collect()
on_disk = tool.INDEX.read_text(encoding="utf-8") if tool.INDEX.is_file() else ""
expected = tool.render(rows)

print(f"[1] 索引存在且覆盖 {len(rows)} 个文件")
check("luts/README.md 存在", bool(on_disk))
missing = [r[0] for r in rows if f"`{r[0]}`" not in on_disk]
check("每个 LUT 都出现在索引里", not missing, f"缺 {len(missing)} 个：{missing[:3]}")

print("[2] 索引内容与重新计算的结果一致")
if on_disk == expected:
    check("逐字一致", True, "无需重新生成")
else:
    a = on_disk.splitlines()
    b = expected.splitlines()
    first = next((i for i in range(min(len(a), len(b))) if a[i] != b[i]), min(len(a), len(b)))
    check("逐字一致", False,
          f"第 {first + 1} 行起不同：磁盘={a[first][:60] if first < len(a) else '<EOF>'} / "
          f"重算={b[first][:60] if first < len(b) else '<EOF>'}；"
          f"请运行 python tools/index_luts.py")

print("[3] 索引的统计口径自洽")
measurable = [r for r in rows if r[2]]
check("绝大多数 LUT 可解析", len(measurable) >= len(rows) - 2,
      f"{len(measurable)}/{len(rows)}")
check("黑白 LUT 的色度接近 0",
      all(r[2]["chroma"] < 0.02 for r in measurable if "bw" in r[0].lower() or "trix" in r[0].lower()),
      "")
check("指标落在合理区间",
      all(0.0 <= r[2]["chroma"] <= 2.5 and -1.0 <= r[2]["warm"] <= 1.0
          and 0.2 <= r[2]["contrast"] <= 1.6 for r in measurable),
      f"最浓 {max(r[2]['chroma'] for r in measurable):.2f}× / "
      f"最暖 {max(r[2]['warm'] for r in measurable):+.2f} / "
      f"反差 {min(r[2]['contrast'] for r in measurable):.2f}~"
      f"{max(r[2]['contrast'] for r in measurable):.2f}")
check("暖度跨越冷暖两侧（指标非退化）",
      min(r[2]["warm"] for r in measurable) < -0.1
      and max(r[2]["warm"] for r in measurable) > 0.1,
      f"{min(r[2]['warm'] for r in measurable):+.2f} ~ "
      f"{max(r[2]['warm'] for r in measurable):+.2f}")

print("\n" + ("LUT 索引校验通过。" if not FAILED else f"失败 {len(FAILED)} 项：{FAILED}"))
sys.exit(1 if FAILED else 0)
