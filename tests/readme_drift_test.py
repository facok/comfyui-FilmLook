# -*- coding: utf-8 -*-
"""README 参数表必须与节点真实输入一致（双向）。

改节点参数却忘了改 README，是这类插件最常见的文档腐化。本测试把 README 里每个节点的
参数表当成契约：表格第一列必须与 INPUT_TYPES 的键**完全一致**（`image` 是连线输入，
不出现在表里）。多一行、少一行、拼错名字都会失败。
"""

import importlib.util
import pathlib
import re
import sys

P = pathlib.Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location(
    "comfyui_FilmLook", P / "__init__.py", submodule_search_locations=[str(P)])
mod = importlib.util.module_from_spec(spec)
sys.modules["comfyui_FilmLook"] = mod
spec.loader.exec_module(mod)

FAILED = []


def check(name, cond, detail=""):
    if cond:
        print(f"  PASS {name}{(' — ' + detail) if detail else ''}")
    else:
        FAILED.append(name)
        print(f"  FAIL {name}{(' — ' + detail) if detail else ''}")


text = (P / "README.md").read_text(encoding="utf-8")
lines = text.splitlines()


def section_body(title):
    """返回 '### <title>' 之后、下一个 '#' 标题之前的行。"""
    for i, ln in enumerate(lines):
        if ln.strip() == f"### {title}":
            body = []
            for nxt in lines[i + 1:]:
                if nxt.startswith("#"):
                    break
                body.append(nxt)
            return body
    return None


def table_first_column(body):
    """取出 markdown 表格里的参数名（跳过表头与分隔行）。"""
    names = []
    for ln in body:
        s = ln.strip()
        if not (s.startswith("|") and s.endswith("|")):
            continue
        cells = [c.strip() for c in s.strip("|").split("|")]
        if not cells:
            continue
        first = cells[0]
        if first in ("参数", "") or set(first) <= set("-: "):
            continue
        names.append(first)
    return names


print("[1] 每个节点在 README 里都有对应章节")
for key, cls in mod.NODE_CLASS_MAPPINGS.items():
    title = mod.NODE_DISPLAY_NAME_MAPPINGS[key]
    body = section_body(title)
    check(f"{title} 有 ### 章节", body is not None)
    if body is None:
        continue

    documented = table_first_column(body)
    actual = [k for k in cls.INPUT_TYPES().get("required", {})
              if not (isinstance(cls.INPUT_TYPES()["required"][k], tuple)
                      and cls.INPUT_TYPES()["required"][k][0] == "IMAGE")]
    actual += list(cls.INPUT_TYPES().get("optional", {}))

    check(f"{title} 参数表与 INPUT_TYPES 一致",
          documented == actual,
          f"文档 {documented} / 实际 {actual}")

print("\n[2] 每个节点都出现在「节点用法」列表里")
usage = None
for i, ln in enumerate(lines):
    if ln.strip().startswith("## 节点用法"):
        usage = "\n".join(lines[i:i + 20])
        break
check("存在「节点用法」章节", usage is not None)
if usage:
    for key, cls in mod.NODE_CLASS_MAPPINGS.items():
        short = mod.NODE_DISPLAY_NAME_MAPPINGS[key].replace("FilmLook ", "").split(" (")[0]
        check(f"用法列表提到「{short}」", f"**{short}**" in usage)

print("\n[3] README 里不该出现已删除的实现（防止文档滞后）")
for stale in ("characteristic_curve", "crosstalk 矩阵", "split_tone", "参数化模型拟合"):
    check(f"未残留「{stale}」", stale not in text)

print("\n[4] 每个 tests/*.py 都要在 README 的测试清单里出现")
for f in sorted((P / "tests").glob("*.py")):
    check(f"README 提到 {f.name}", f.name in text)

print("\n[5] 关键事实必须被写明")
for must in ("presets/LICENSE.md", "preset_character_test.py", "夜景暖调"):
    check(f"README 提到 {must}", must in text)

print("\n" + ("README 与实现一致。" if not FAILED else f"失败 {len(FAILED)} 项：{FAILED}"))
sys.exit(1 if FAILED else 0)
