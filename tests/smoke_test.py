# -*- coding: utf-8 -*-
"""FilmLook 冒烟测试：合成测试图跑全部 5 个节点，检查形状/dtype/值域/NaN。
用法：python tests/smoke_test.py（无需 ComfyUI 环境，仅需 torch/numpy/Pillow）。"""

import importlib.util
import pathlib
import struct
import sys
import tempfile
import zlib

import numpy as np
import torch
from PIL import Image

PKG_DIR = pathlib.Path(__file__).resolve().parent.parent


def load_plugin():
    spec = importlib.util.spec_from_file_location(
        "comfyui_FilmLook", PKG_DIR / "__init__.py", submodule_search_locations=[str(PKG_DIR)])
    mod = importlib.util.module_from_spec(spec)
    sys.modules["comfyui_FilmLook"] = mod
    spec.loader.exec_module(mod)
    return mod


def make_test_image():
    img = torch.zeros(1, 256, 256, 3)
    img[0, :128, :128] = torch.tensor([0.98, 0.97, 0.95])   # 高光
    img[0, :128, 128:] = 0.46                                # 中灰
    img[0, 128:, :128] = torch.tensor([0.87, 0.66, 0.55])    # 肤色
    img[0, 128:, 128:] = 0.02                                # 暗部
    img[0, 20:28, 20:28] = 1.0                               # 极亮小点（触发光晕）
    return img


def check(name, out, ref):
    assert isinstance(out, torch.Tensor), f"{name}: 输出不是 tensor"
    assert out.shape == ref.shape, f"{name}: 形状 {tuple(out.shape)} != {tuple(ref.shape)}"
    assert out.dtype == torch.float32, f"{name}: dtype {out.dtype}"
    assert not torch.isnan(out).any(), f"{name}: 含 NaN"
    assert float(out.min()) >= 0.0 and float(out.max()) <= 1.0, \
        f"{name}: 值域 [{float(out.min())}, {float(out.max())}] 越界"
    print(f"  PASS {name}: shape={tuple(out.shape)} range=[{float(out.min()):.4f}, {float(out.max()):.4f}]")


IDENTITY_CUBE = """TITLE "identity"
LUT_3D_SIZE 2
0 0 0
1 0 0
0 1 0
1 1 0
0 0 1
1 0 1
0 1 1
1 1 1
"""

GRAY_CUBE = """TITLE "gray"
LUT_3D_SIZE 2
0.5 0.5 0.5
0.5 0.5 0.5
0.5 0.5 0.5
0.5 0.5 0.5
0.5 0.5 0.5
0.5 0.5 0.5
0.5 0.5 0.5
0.5 0.5 0.5
"""

GRAY_1D_CUBE = """TITLE "gray1d"
LUT_1D_SIZE 2
0.5 0.5 0.5
0.5 0.5 0.5
"""


def make_haldclut(level, fn):
    """按 ImageMagick hald 布局生成 HaldCLUT：线性序号 i=(b*S+g)*S+r，红最快蓝最慢。"""
    s = level * level
    side = level * s
    idx = np.arange(side * side, dtype=np.int64)
    r = (idx % s) / (s - 1)
    g = ((idx // s) % s) / (s - 1)
    b = (idx // (s * s)) / (s - 1)
    return np.stack(fn(r, g, b), axis=-1).reshape(side, side, 3)


def write_png16(path, arr):
    """写 16-bit RGB PNG（filter 0）。PIL 不能写 16-bit RGB，用 stdlib 手写。"""
    h, w = arr.shape[:2]
    data = (arr * 65535).round().astype(np.uint16)
    raw = b"".join(b"\x00" + row.astype(">u2").tobytes() for row in data)

    def chunk(tag, payload):
        c = tag + payload
        return struct.pack(">I", len(payload)) + c + struct.pack(">I", zlib.crc32(c))

    png = (b"\x89PNG\r\n\x1a\n"
           + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 16, 2, 0, 0, 0))
           + chunk(b"IDAT", zlib.compress(raw))
           + chunk(b"IEND", b""))
    pathlib.Path(path).write_bytes(png)


def haldclut_tests(mod, node, img, td):
    identity = lambda r, g, b: (r, g, b)
    # 8 级 16-bit identity：往返误差须 <1e-3
    p16 = pathlib.Path(td, "identity16.png")
    write_png16(p16, make_haldclut(8, identity))
    out = node.execute(img, "自定义路径", 1.0, str(p16))[0]
    check("HaldCLUT[8级16bit恒等]", out, img)
    err = float((out - img).abs().max())
    assert err < 1e-3, f"8 级 16-bit 恒等 HaldCLUT 误差 {err} 应 <1e-3"
    print(f"  PASS HaldCLUT[8级16bit恒等] 往返误差 {err:.2e}")
    # 4 级 8-bit identity：节点值恰好在 8-bit 网格上，同样须 <1e-3
    p8 = pathlib.Path(td, "identity8.png")
    Image.fromarray((make_haldclut(4, identity) * 255).round().astype(np.uint8)).save(p8)
    out = node.execute(img, "自定义路径", 1.0, str(p8))[0]
    err = float((out - img).abs().max())
    assert err < 1e-3, f"4 级 8-bit 恒等 HaldCLUT 误差 {err} 应 <1e-3"
    print(f"  PASS HaldCLUT[4级8bit恒等] 往返误差 {err:.2e}")
    # 蓝色通道压成纯灰：验证非平凡映射
    pgray = pathlib.Path(td, "bluegray.png")
    Image.fromarray((make_haldclut(4, lambda r, g, b: (r, g, np.full_like(b, 0.5))) * 255)
                    .round().astype(np.uint8)).save(pgray)
    out = node.execute(img, "自定义路径", 1.0, str(pgray))[0]
    check("HaldCLUT[蓝色变灰]", out, img)
    assert float((out[..., 2] - 0.5).abs().max()) < 1e-2, "蓝色通道应全部变为 0.5"
    assert float((out[..., :2] - img[..., :2]).abs().max()) < 1e-3, "红绿通道应保持不变"
    print("  PASS HaldCLUT[蓝色变灰] B→0.5，R/G 不变")
    # 非正方形 PNG 与不支持扩展名的错误路径
    bad = pathlib.Path(td, "bad.png")
    Image.fromarray(np.zeros((8, 16, 3), dtype=np.uint8)).save(bad)
    try:
        node.execute(img, "自定义路径", 1.0, str(bad))
        raise AssertionError("非正方形 PNG 应报错")
    except RuntimeError as e:
        print(f"  PASS HaldCLUT[非正方形报错]: {e}")
    txt = pathlib.Path(td, "fake.txt")
    txt.write_text("not a lut", encoding="ascii")
    try:
        node.execute(img, "自定义路径", 1.0, str(txt))
        raise AssertionError(".txt 应报不支持")
    except RuntimeError as e:
        print(f"  PASS LUT[不支持扩展名报错]: {e}")
    # 下拉框应列出 luts/ 目录中的 .png
    probe = pathlib.Path(mod.color_nodes.LUTS_DIR) / "zz_smoke_probe.png"
    try:
        Image.fromarray((make_haldclut(2, identity) * 255).round().astype(np.uint8)).save(probe)
        luts = mod.FilmLookLUT.INPUT_TYPES()["required"]["lut"][0]
        assert "zz_smoke_probe.png" in luts, f"下拉框未列出 .png: {luts[-3:]}"
        out = node.execute(img, "zz_smoke_probe.png", 1.0)[0]
        check("HaldCLUT[下拉框选择]", out, img)
        print("  PASS HaldCLUT[下拉框列出并加载 luts/ 内 .png]")
    finally:
        probe.unlink(missing_ok=True)


def main():
    mod = load_plugin()
    img = make_test_image()
    batch = torch.cat([img, img * 0.5], dim=0)

    print("[1] FilmLookColor（全部预设 + 批次）")
    node = mod.FilmLookColor()
    for preset in mod.FilmLookColor.INPUT_TYPES()["required"]["preset"][0]:
        out = node.execute(img, preset, 0.85, True, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0)[0]
        check(f"Color[{preset}]", out, img)
    check("Color[手动参数]", node.execute(img, "自定义", 1.0, True, 0.3, 1.2, 1.3, 0.2, -0.1, 0.0, 1.0)[0], img)
    check("Color[批次x2]", node.execute(batch, "暖阳金调 (Kodak Gold)", 0.85, True, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0)[0], batch)
    check("Color[关智能曝光]", node.execute(img, "冷调电影感 (Vision3 500T)", 1.0, False, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0)[0], img)
    check("Color[亮度保持]", node.execute(img, "浓郁鲜艳 (Ektar)", 1.0, True, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0)[0], img)
    check("Color[风格浓度x2]", node.execute(img, "暖阳金调 (Kodak Gold)", 1.0, True, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 2.0)[0], img)

    print("[2] FilmLookHalation（自动/手动阈值 + 批次）")
    node = mod.FilmLookHalation()
    for radius in ["轻微", "标准", "明显", "夸张"]:
        check(f"Halation[{radius}]", node.execute(img, 0.5, radius, True, 0.85)[0], img)
    check("Halation[手动阈值]", node.execute(img, 0.8, "标准", False, 0.5)[0], img)
    check("Halation[批次x2]", node.execute(batch, 0.5, "标准", True, 0.85)[0], batch)

    print("[2b] FilmLookLens（柔光/暗角/关闭）")
    node = mod.FilmLookLens()
    check("Lens[默认]", node.execute(img, 0.25, 0.25)[0], img)
    assert torch.equal(node.execute(img, 0.0, 0.0)[0], img), "Lens 全关应严格返回原图"
    print("  PASS Lens[全关恒等]")
    dark = node.execute(img, 0.0, 0.8)[0]
    assert float(dark[0, 5, 5].mean()) < float(img[0, 5, 5].mean()), "暗角应压暗边角"
    assert abs(float(dark[0, 128, 128].mean()) - float(img[0, 128, 128].mean())) < 0.02, "暗角不应影响中心"
    print("  PASS Lens[暗角只压边角]")
    glow = node.execute(img, 0.8, 0.0)[0]
    assert float(glow[0, 128, 128].mean()) > float(img[0, 128, 128].mean()), "柔光应提亮高光区周围"
    print("  PASS Lens[柔光提亮高光周围]")

    print("[3] FilmLookLUT（3D 恒等 / 3D 纯灰 / 1D 曲线 / 错误路径）")
    node = mod.FilmLookLUT()
    luts = mod.FilmLookLUT.INPUT_TYPES()["required"]["lut"][0]
    assert "自定义路径" in luts, f"LUT 下拉缺少自定义路径: {luts}"
    print(f"  PASS LUT 下拉框内容: {luts}")
    with tempfile.TemporaryDirectory() as td:
        p_id = pathlib.Path(td, "identity.cube")
        p_id.write_text(IDENTITY_CUBE, encoding="ascii")
        p_gray = pathlib.Path(td, "gray.cube")
        p_gray.write_text(GRAY_CUBE, encoding="ascii")
        p_1d = pathlib.Path(td, "gray1d.cube")
        p_1d.write_text(GRAY_1D_CUBE, encoding="ascii")
        out0 = node.execute(img, "自定义路径", 0.0, str(p_id))[0]
        assert torch.equal(out0, img), "strength=0 应严格返回原图"
        print("  PASS LUT[strength=0 严格恒等]")
        out1 = node.execute(img, "自定义路径", 1.0, str(p_id))[0]
        check("LUT[3D恒等 strength=1]", out1, img)
        assert float((out1 - img).abs().max()) < 1e-4, "恒等 LUT 在 strength=1 时应近似原图"
        outg = node.execute(img, "自定义路径", 1.0, str(p_gray))[0]
        check("LUT[3D纯灰]", outg, img)
        assert float((outg - 0.5).abs().max()) < 1e-4, "纯灰 LUT 应输出全 0.5"
        print("  PASS LUT[3D纯灰 -> 全 0.5]")
        check("LUT[1D曲线]", node.execute(img, "自定义路径", 1.0, str(p_1d))[0], img)
        try:
            node.execute(img, "自定义路径", 1.0, str(pathlib.Path(td, "missing.cube")))
            raise AssertionError("缺失 LUT 文件应报错")
        except RuntimeError as e:
            print(f"  PASS LUT[缺失文件报错]: {e}")
        haldclut_tests(mod, node, img, td)

    print("[4] FilmLookGrain（silvergrain 检测）")
    try:
        import silvergrain  # noqa: F401
        has_sg = True
    except ImportError:
        has_sg = False
    node = mod.FilmLookGrain()
    if has_sg:
        out = node.execute(img, "标准 (ISO 400)", "亮度颗粒 (保色彩)", 1.0, "速度优先", 0.3, 0)[0]
        check("Grain[速度优先]", out, img)
    else:
        try:
            node.execute(img, "标准 (ISO 400)", "亮度颗粒 (保色彩)", 1.0, "速度优先", 0.3, 0)
            raise AssertionError("未安装 silvergrain 时应报 RuntimeError")
        except RuntimeError as e:
            assert "silvergrain" in str(e) and "pip install" in str(e)
            print(f"  PASS Grain[未安装报错]:\n    {str(e).splitlines()[0]}")

    print("[5] FilmLookOneClick")
    node = mod.FilmLookOneClick()
    if has_sg:
        check("OneClick", node.execute(img, "清新人像 (Portra 400)", "极细 (ISO 100)", 0.85, 0.4, True, 0.3, 1.0, 0.2, 0.2, 0)[0], img)
    else:
        try:
            node.execute(img, "清新人像 (Portra 400)", "标准 (ISO 400)", 0.85, 0.4, True, 0.3, 1.0, 0.2, 0.2, 0)
            raise AssertionError("未安装 silvergrain 时一键节点应报 RuntimeError")
        except RuntimeError as e:
            assert "silvergrain" in str(e)
            print("  PASS OneClick[未安装 silvergrain 报错，信息与颗粒节点一致]")

    print("\n全部冒烟测试通过。")


if __name__ == "__main__":
    main()
