# -*- coding: utf-8 -*-
"""节点契约测试：注册表、签名、默认值、返回声明的一致性。

这类错误不会让节点跑挂，但会让界面表现错乱（比如默认值落在 min/max 之外、combo 的
默认项不在选项里、execute 参数名与 INPUT_TYPES 键对不上导致 ComfyUI 传参失败）。
放成常驻测试，改节点声明时自动兜底。
"""

import importlib.util
import inspect
import pathlib
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


print("[1] 注册表完整性")
maps = mod.NODE_CLASS_MAPPINGS
names = mod.NODE_DISPLAY_NAME_MAPPINGS
check("NODE_CLASS_MAPPINGS 非空且 ≥5 个节点", len(maps) >= 5, f"{len(maps)} 个")
check("每个节点都有中文显示名", set(maps) == set(names) and all(names.values()),
      f"缺 {set(maps) - set(names)}")
check("显示名无重复", len(set(names.values())) == len(names))
check("__all__ 导出注册表", set(mod.__all__) == {"NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"})

for key, cls in maps.items():
    print(f"\n[{key}] {names[key]}")
    it = cls.INPUT_TYPES()
    required = it.get("required", {})
    optional = it.get("optional", {})
    all_inputs = {**required, **optional}

    # 1) 签名与输入声明一致
    fn = getattr(cls, cls.FUNCTION, None)
    check("FUNCTION 指向的方法存在", callable(fn), cls.FUNCTION)
    params = [p for p in inspect.signature(fn).parameters if p != "self"]
    check("execute 参数与 INPUT_TYPES 完全对应",
          set(params) == set(all_inputs),
          f"多余 {set(params) - set(all_inputs)} / 缺少 {set(all_inputs) - set(params)}")
    check("execute 参数顺序与 INPUT_TYPES 一致",
          params == list(all_inputs), "")
    required_params = [p for p in params
                       if inspect.signature(fn).parameters[p].default is inspect.Parameter.empty]
    check("必填参数都有默认值（ComfyUI 允许省略 optional）",
          set(required_params) <= set(all_inputs), f"{required_params}")

    # 2) 返回声明
    check("RETURN_TYPES 与 RETURN_NAMES 等长",
          len(cls.RETURN_TYPES) == len(cls.RETURN_NAMES),
          f"{cls.RETURN_TYPES} vs {cls.RETURN_NAMES}")
    check("分类已填写", bool(getattr(cls, "CATEGORY", "")))

    # 3) 每个输入的默认值都要合法
    for name, spec_ in all_inputs.items():
        # ComfyUI 允许 ("IMAGE",) 这种只有类型、无选项的写法（socket-only 输入）
        if not (isinstance(spec_, tuple) and 1 <= len(spec_) <= 2):
            check(f"{name} 输入声明格式", False, str(spec_)[:40])
            continue
        kind = spec_[0]
        opts = (spec_[1] if len(spec_) > 1 else None) or {}
        default = opts.get("default", None)
        if isinstance(kind, (list, tuple)):                     # combo
            check(f"{name}: 默认项在选项里", default is None or default in kind,
                  f"default={default!r}")
            check(f"{name}: 选项非空", len(kind) > 0)
        elif kind in ("INT", "FLOAT"):                          # 数值
            lo, hi = opts.get("min"), opts.get("max")
            ok = default is None or (lo is None or default >= lo) and (hi is None or default <= hi)
            check(f"{name}: 默认值在 [min,max] 内", ok,
                  f"default={default} range=[{lo},{hi}]")
            check(f"{name}: 声明了 min/max", lo is not None and hi is not None)
        elif kind == "BOOLEAN":
            check(f"{name}: 默认值是 bool", isinstance(default, bool), f"{default!r}")

print("\n[2] 节点名与分类命名空间")
for key, cls in maps.items():
    check(f"{key} 以 FilmLook 开头", key.startswith("FilmLook_"))
    check(f"{key} 分类以 FilmLook 开头", cls.CATEGORY.startswith("FilmLook"), cls.CATEGORY)

print("\n" + ("全部节点契约校验通过。" if not FAILED else f"失败 {len(FAILED)} 项：{FAILED}"))
sys.exit(1 if FAILED else 0)
