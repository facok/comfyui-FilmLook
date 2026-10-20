# 下载辅助：把 FetchURL 保存的抓取结果转成合法 .cube 并用插件的 parse_cube 验证
# 用法：python tests/fetch_result_to_cube.py <fetch结果txt> <输出cube路径>
import importlib.util
import pathlib
import sys

src, dst = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
text = src.read_text(encoding="utf-8")
lines = text.splitlines()
# 跳过 FetchURL 的说明头，找到 .cube 内容起点
start = next(i for i, ln in enumerate(lines) if ln.startswith("TITLE") or ln.startswith("LUT"))
body = "\n".join(ln.replace("\\_", "_") for ln in lines[start:]) + "\n"
dst.parent.mkdir(parents=True, exist_ok=True)
dst.write_text(body, encoding="ascii")

pkg = pathlib.Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location(
    "comfyui_FilmLook", pkg / "__init__.py", submodule_search_locations=[str(pkg)])
mod = importlib.util.module_from_spec(spec)
sys.modules["comfyui_FilmLook"] = mod
spec.loader.exec_module(mod)
lut = mod.film_core.parse_cube(str(dst))
rows = lut["size"] ** 3
print(f"OK {dst.name}: type={lut['type']} size={lut['size']} rows={rows} "
      f"range=[{float(lut['table'].min()):.4f},{float(lut['table'].max()):.4f}]")
