# -*- coding: utf-8 -*-
"""FilmLook 共享数学：色彩空间转换、特性曲线、串扰矩阵、分离色调、halation、.cube 解析。"""

import collections
import math
import os
import struct
import zlib

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

_LUMA = (0.2126, 0.7152, 0.0722)


def _luma_w(device, dtype):
    return torch.tensor(_LUMA, device=device, dtype=dtype).view(1, 1, 1, 3)


def pick(kind, value, options):
    """按键取值，键不存在时给出**带可选值**的报错。

    裸 KeyError 只会说 "'xxx'"，用户（或三个月后的你）根本不知道有效值是什么。
    这种值只可能来自旧工作流或直接调 API——下拉框本身有 ComfyUI 校验。
    """
    try:
        return options[value]
    except KeyError:
        valid = " / ".join(str(k) for k in options)
        raise ValueError(
            f"{kind}“{value}”不在可选范围内。可选：{valid}。"
            f"如果这是旧工作流里保存的值，重新选一次即可。") from None


def sanitize(image):
    """入口清洗：NaN→0、+inf→1、-inf→0。对有限值完全恒等，正常路径无副作用。

    为什么需要：上游节点偶发一个 NaN 像素（VAE 解码、除零、mask 广播出错）时，
    三线性插值和后面的高斯模糊会把这一点污染扩散到**整幅图**，用户看到的是全黑/全白
    却完全不知道问题出在上游。在入口挡一次，代价是一次逐元素操作。
    """
    return torch.nan_to_num(image.float(), nan=0.0, posinf=1.0, neginf=0.0)


def split_alpha(image):
    if image.shape[-1] == 4:
        return image[..., :3], image[..., 3:4]
    return image, None


def _join_alpha(rgb, alpha):
    if alpha is None:
        return rgb
    return torch.cat([rgb, alpha], dim=-1)


def srgb_to_linear(x):
    x = x.clamp_min(0.0)
    return torch.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(x):
    x = x.clamp_min(0.0)
    # pow 分支钳离 0：正向取值不变（x=0 走线性分支），避免 0^(1/2.4) 反向梯度 inf*0=NaN
    return torch.where(x <= 0.0031308, x * 12.92, 1.055 * x.clamp_min(1e-12) ** (1.0 / 2.4) - 0.055)


def luminance(x):
    return (x * _luma_w(x.device, x.dtype)).sum(dim=-1, keepdim=True)


def smoothstep(e0, e1, x):
    t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _frame_samples(x, max_samples=1_000_000):
    """(B,H,W,1) → (B,N)：逐帧展平，样本数超上限时等间隔抽样。

    torch.quantile 对单次归约维度有 2^24 元素的上限，单帧超过约 1677 万像素
    （8K、大横幅）会直接抛 "quantile() input tensor is too large"；抽样顺带把大图的
    中位数/分位数计算从数百毫秒压到个位数毫秒。
    """
    flat = x.reshape(x.shape[0], -1)
    if flat.shape[1] > max_samples:
        flat = flat[:, :: math.ceil(flat.shape[1] / max_samples)]
    return flat


def _percentile_per_frame(x, q, max_samples=1_000_000):
    """逐帧百分位：x (B,H,W,1) → (B,1,1,1)，避免整批共享统计量导致曝光/阈值互相拖累。"""
    return torch.quantile(_frame_samples(x, max_samples), q, dim=1).view(-1, 1, 1, 1)


def auto_exposure_gain(lin, target=0.18):
    """逐帧曝光增益：(B,H,W,C) → (B,1,1,1)。半强度中位数校正（开根号）+ 温和区间
    [0.75, 1.4] + 高光保护：AI 出图通常曝光正常，只纠明显偏差，不做激进重锚定。"""
    lum = luminance(lin)
    flat = _frame_samples(lum)
    med = flat.median(dim=1).values.clamp_min(1e-4)
    gain = (target / med).sqrt().clamp(0.75, 1.4)
    p95 = torch.quantile(flat, 0.95, dim=1).clamp_min(1e-4)
    cap = (0.9 / p95).clamp(1.0, 2.0)
    return torch.minimum(gain, cap).view(-1, 1, 1, 1)



def apply_contrast(lin, contrast, pivot=0.18):
    if abs(contrast - 1.0) < 1e-6:
        return lin
    return pivot * (lin.clamp_min(1e-6) / pivot) ** contrast


def apply_vibrance(lin, saturation):
    """自然饱和度：降饱和是全线等比；升饱和只加在"颜色本来就淡"的像素上，
    已经很鲜艳的地方几乎不动，所以肤色不容易被推爆。

    （这里不再保留 tensor 分支——那是给已删除的参数拟合脚本求梯度用的。）
    """
    if abs(float(saturation) - 1.0) < 1e-6:
        return lin
    lum = luminance(lin)
    if saturation < 1.0:
        return lum + (lin - lum) * float(saturation)
    spread = lin.amax(dim=-1, keepdim=True) - lin.amin(dim=-1, keepdim=True)
    boost = 1.0 + (float(saturation) - 1.0) * (1.0 - spread).clamp(0.0, 1.0)
    return lum + (lin - lum) * boost

def apply_temp_tint(lin, temperature, tint, amount=0.30):
    """色温 / 色调：线性域亮度守恒增益（白平衡语义，与 Lightroom 同向）。

    temperature > 0 偏暖（抬 R、压 B），< 0 偏冷；tint > 0 偏品红（压 G），< 0 偏绿。
    增益写成 (1±amount)**t 而非 1±amount·t：正负行程严格互反（+1 与 -1 互为倒数），
    否则暖/冷两侧的色偏差不自洽（1.12 与 0.88 并不是一对倒数）。
    amount 定的是满行程尺度：0.30 时 R/B≈1.86，按黑体辐射标定，等效白点由 D65(6500K)
    移到约 4620K（暖）/ 11600K（冷），即 ±65 mired 左右——这是"色温"控件该有的量级；
    原先 0.12 只有 ±25 mired（6500K→5600K），滑杆大半行程几乎看不出变化。
    """
    t = float(temperature)
    n = float(tint)
    if abs(t) < 1e-6 and abs(n) < 1e-6:
        return lin
    g = torch.tensor([(1.0 + amount) ** t, (1.0 - amount) ** n, (1.0 - amount) ** t],
                     device=lin.device, dtype=lin.dtype).view(1, 1, 1, 3)
    g = g / (g * _luma_w(lin.device, lin.dtype)).sum()  # 归一化：中灰亮度不变
    return lin * g


def apply_film_color(image, stock, *, strength=0.85, auto_exposure=True, brightness=0.0,
                     contrast=1.0, saturation=1.0, temperature=0.0, tint=0.0,
                     preserve_luminance=0.0, style_boost=1.0):
    image = sanitize(image)
    """胶片调色主管线：sRGB→线性 → 智能曝光/亮度 → 回显示域 → 查 3D LUT →
    线性域做对比度 / 自然饱和度 / 色温色调 → 回 sRGB → 保留亮度 → 按强度混合。

    stock 需含 "table"（parse_cube 的解析结果，None 表示中性直通）与 "monochrome"。
    预设即 LUT：不再用参数化模型拟合，因此输出与参考 LUT 逐点一致（见 film_stocks.py）。

    preserve_luminance: 0=允许 LUT 自由改变亮度；1=输出亮度分布完全贴回原图。
    style_boost: 风格浓度，1=忠实 LUT，>1 把 LUT 对原图的偏离等比放大（超真实胶片）。
    """
    image = image.float()
    if float(strength) * float(style_boost) <= 0.0:
        # 强度 0 = 原图。黑白预设尤其需要这道闸门：它的混合基准是灰度原图，
        # 不给提前返回的话 strength=0 会返回一张灰度图而不是彩色原图。
        return image
    rgb, alpha = split_alpha(image)
    src = rgb.clamp(0.0, 1.0)

    # 曝光类调整必须在查表之前：LUT 是按"曝光正常"的画面标定的
    lin = srgb_to_linear(src)
    if auto_exposure:
        lin = lin * auto_exposure_gain(lin)
    if abs(brightness) > 1e-6:
        lin = lin * float(2.0 ** brightness)
    work = linear_to_srgb(lin).clamp(0.0, 1.0)

    table = stock.get("table")
    if table is not None:
        work = apply_cube(work, table, 1.0)

    # 观感调整在 LUT 之后：它们是"往成片上再加的调色"
    out_lin = srgb_to_linear(work)
    out_lin = apply_contrast(out_lin, contrast)
    out_lin = apply_vibrance(out_lin, saturation)
    out_lin = apply_temp_tint(out_lin, temperature, tint)
    out = linear_to_srgb(out_lin).clamp(0.0, 1.0)
    if preserve_luminance > 0.0:
        eps = 1e-4
        ratio = (luminance(src) + eps) / (luminance(out) + eps)
        matched = (out * ratio).clamp(0.0, 1.0)
        out = out.lerp(matched, float(preserve_luminance))
    blend_src = src
    if stock.get("monochrome"):
        # 黑白预设的强度混合必须在灰度域进行，否则剩余的原图色彩会把黑白照染回彩色
        blend_src = linear_to_srgb(luminance(srgb_to_linear(src))).expand_as(src)
    out = blend_src + (out - blend_src) * (float(strength) * float(style_boost))
    return _join_alpha(out.clamp(0.0, 1.0), alpha)


def _gaussian_blur(x, sigma):
    """可分离高斯模糊。半径按各轴长度夹紧：F.pad(mode="reflect") 要求 pad < 该轴长度，
    否则细长图（4096x16、512x1）和超小图（1x1）会直接抛 "Padding size should be less
    than the corresponding input dimension"。夹紧等价于截断核，对正常尺寸无任何影响。"""
    if sigma < 0.3:
        return x
    c = x.shape[-1]
    h, w = x.shape[1], x.shape[2]
    y = x.permute(0, 3, 1, 2)
    for axis, length in ((3, w), (2, h)):
        radius = max(0, min(int(math.ceil(sigma * 3.0)), length - 1))
        xs = torch.arange(-radius, radius + 1, device=x.device, dtype=x.dtype)
        k = torch.exp(-0.5 * (xs / sigma) ** 2)
        k = k / k.sum()
        if axis == 3:  # 横向
            y = F.conv2d(F.pad(y, (radius, radius, 0, 0), mode="reflect"),
                         k.view(1, 1, 1, -1).expand(c, 1, 1, -1), groups=c)
        else:          # 纵向
            y = F.conv2d(F.pad(y, (0, 0, radius, radius), mode="reflect"),
                         k.view(1, 1, -1, 1).expand(c, 1, -1, 1), groups=c)
    return y.permute(0, 2, 3, 1)


def apply_halation(image, *, strength=0.5, scales=(0.01, 0.02), auto_threshold=True,
                   threshold=0.85, tint=(1.0, 0.45, 0.2)):
    """高光红橙光晕：线性域亮度阈值(软膝)取高光 → 多尺度高斯模糊 → 染成暖色 → 滤色混合。"""
    image = sanitize(image)
    if float(strength) <= 0.0:
        return image  # 强度 0 = 原图，同时省掉多尺度模糊的开销
    image = image.float()
    rgb, alpha = split_alpha(image)
    lin = srgb_to_linear(rgb.clamp(0.0, 1.0))
    lum = luminance(lin)
    if auto_threshold:
        thr = _percentile_per_frame(lum, 0.98)  # (B,1,1,1) 逐帧阈值
    else:
        thr = srgb_to_linear(torch.tensor(float(threshold), device=lin.device, dtype=lin.dtype))
    # 蒙版在阈值处达到 1（软膝下沿取阈值的 70%）：保证恰好等于阈值/顶格 1.0 的高光也有完整光晕
    lo = thr * 0.7
    mask = smoothstep(lo, torch.clamp(thr, min=lo + 1e-3), lum)
    hi = lin * mask
    # 尺度按短边统一：原先用 height，横图里 height 恰是短边、竖图里却是长边，
    # 同一档"标准"在竖图上会给出约 1.8 倍大的光晕。
    scale = min(image.shape[1], image.shape[2])
    weights = [0.5 ** i for i in range(len(scales))]
    halo = sum(w * _gaussian_blur(hi, s * scale) for w, s in zip(weights, scales)) / sum(weights)
    tint_lin = srgb_to_linear(torch.tensor(tint, device=lin.device, dtype=lin.dtype).view(1, 1, 1, 3))
    halo_rgb = luminance(halo) * tint_lin
    out = 1.0 - (1.0 - lin) * (1.0 - (halo_rgb * float(strength)).clamp(0.0, 1.0))
    out = linear_to_srgb(out).clamp(0.0, 1.0)
    return _join_alpha(out, alpha)


def apply_diffusion(image, *, strength=0.2):
    image = sanitize(image)
    """柔光（黑柔镜效果）：高光低阈值软膝提取 → 高斯模糊 → 中性滤色叠加。
    与 halation 的区别：不着色、阈值更低、作用更弥散，压低微反差带来柔焦感。"""
    if strength <= 0.0:
        return image
    image = image.float()
    rgb, alpha = split_alpha(image)
    lin = srgb_to_linear(rgb.clamp(0.0, 1.0))
    lum = luminance(lin)
    mask = smoothstep(0.25, 0.75, lum)
    hi = lin * mask
    scale = min(image.shape[1], image.shape[2])  # 同 halation：按短边统一，横竖图一致
    glow = (_gaussian_blur(hi, 0.012 * scale) + 0.5 * _gaussian_blur(hi, 0.03 * scale)) / 1.5
    out = 1.0 - (1.0 - lin) * (1.0 - (glow * float(strength)).clamp(0.0, 1.0))
    out = linear_to_srgb(out).clamp(0.0, 1.0)
    return _join_alpha(out, alpha)


def apply_vignette(image, *, strength=0.25):
    image = sanitize(image)
    """暗角：以画面中心为圆心的径向压暗，线性域乘法，形状固定只需一个强度。"""
    if strength <= 0.0:
        return image
    image = image.float()
    rgb, alpha = split_alpha(image)
    b, h, w, _ = image.shape
    # 用像素中心归一化：linspace(-1,1,1) 在单像素轴上会返回 -1，把整幅图当成角落；
    # 这里 n=1 时得到 0（正中），n>1 时接近 ±1。
    ys = (torch.arange(h, device=image.device, dtype=image.dtype) + 0.5) * (2.0 / h) - 1.0
    xs = (torch.arange(w, device=image.device, dtype=image.dtype) + 0.5) * (2.0 / w) - 1.0
    dist = (xs.view(1, 1, w, 1) ** 2 + ys.view(1, h, 1, 1) ** 2).sqrt() / math.sqrt(2.0)
    falloff = 1.0 - float(strength) * smoothstep(0.45, 1.0, dist)
    lin = srgb_to_linear(rgb.clamp(0.0, 1.0)) * falloff
    return _join_alpha(linear_to_srgb(lin).clamp(0.0, 1.0), alpha)


def _lut_bytes(lut):
    table = lut["table"]
    return table.numel() * table.element_size()


_LUT_CACHE = collections.OrderedDict()   # (绝对路径, mtime_ns, 文件大小) → parsed
_LUT_CACHE_MAX_BYTES = 256 * 2 ** 20
_LUT_CACHE_MAX_ENTRIES = 8


def _cached_parse(path, parser):
    """按 (绝对路径, mtime, 文件大小) 缓存 LUT 解析结果。

    .cube 解析只要几毫秒，但 16-bit HaldCLUT 的 PNG 反滤波是逐字节 Python 循环：
    1728² 约 12 秒、4096² 约 67 秒，不缓存等于每出一张图就重付一遍。文件被替换时
    mtime/size 会变化，缓存自动失效；总字节数超预算时按 LRU 淘汰，避免大 LUT 撑爆内存。
    """
    try:
        st = os.stat(path)
    except OSError:
        return parser(path)  # 交给 parser 抛出统一的"找不到 LUT 文件"
    key = (os.path.abspath(path), st.st_mtime_ns, st.st_size)
    if key in _LUT_CACHE:
        _LUT_CACHE.move_to_end(key)
        return _LUT_CACHE[key]
    lut = parser(path)
    if _lut_bytes(lut) <= _LUT_CACHE_MAX_BYTES:
        _LUT_CACHE[key] = lut
        total = sum(_lut_bytes(v) for v in _LUT_CACHE.values())
        while _LUT_CACHE and (total > _LUT_CACHE_MAX_BYTES or len(_LUT_CACHE) > _LUT_CACHE_MAX_ENTRIES):
            _, old = _LUT_CACHE.popitem(last=False)
            total -= _lut_bytes(old)
    return lut


def parse_cube(path):
    """解析 .cube 文件，支持 LUT_1D_SIZE / LUT_3D_SIZE、TITLE、DOMAIN_MIN/MAX、# 注释。"""
    return _cached_parse(path, _parse_cube_file)


def _parse_cube_file(path):
    if not os.path.isfile(path):
        raise RuntimeError(f"找不到 LUT 文件：{path}")
    size_1d = None
    size_3d = None
    domain_min = [0.0, 0.0, 0.0]
    domain_max = [1.0, 1.0, 1.0]
    rows = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for lineno, raw in enumerate(f, 1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            key = line.split(None, 1)[0].upper()
            if key == "TITLE":
                continue
            if key == "LUT_1D_SIZE":
                size_1d = int(line.split()[1])
                continue
            if key == "LUT_3D_SIZE":
                size_3d = int(line.split()[1])
                continue
            if key == "DOMAIN_MIN":
                domain_min = [float(v) for v in line.split()[1:4]]
                continue
            if key == "DOMAIN_MAX":
                domain_max = [float(v) for v in line.split()[1:4]]
                continue
            parts = line.split()
            if len(parts) >= 3:
                try:
                    rows.append((float(parts[0]), float(parts[1]), float(parts[2])))
                except ValueError:
                    raise RuntimeError(f"LUT 文件第 {lineno} 行无法解析：{line}")
    data = torch.tensor(rows, dtype=torch.float32)
    if size_3d is not None:
        if data.shape[0] != size_3d ** 3:
            raise RuntimeError(f"LUT_3D_SIZE={size_3d} 应有 {size_3d ** 3} 行数据，实际 {data.shape[0]} 行")
        # .cube 数据 R 变化最快，reshape 后维度为 [b, g, r, 通道]
        return {"type": "3d", "size": size_3d, "table": data.view(size_3d, size_3d, size_3d, 3),
                "domain_min": domain_min, "domain_max": domain_max}
    if size_1d is not None:
        if data.shape[0] != size_1d:
            raise RuntimeError(f"LUT_1D_SIZE={size_1d} 应有 {size_1d} 行数据，实际 {data.shape[0]} 行")
        return {"type": "1d", "size": size_1d, "table": data.view(size_1d, 3),
                "domain_min": domain_min, "domain_max": domain_max}
    raise RuntimeError(f"LUT 文件缺少 LUT_1D_SIZE / LUT_3D_SIZE 声明：{path}")


def _unfilter_png(raw, height, stride, bpp):
    """PNG 扫描线反滤波（filter 0-4）。Sub/Up 用 numpy 向量化，Average/Paeth 逐字节递推。"""
    out = np.zeros((height, stride), dtype=np.uint8)
    prev = np.zeros(stride, dtype=np.uint8)
    p = 0
    for y in range(height):
        f = raw[p]
        p += 1
        line = np.frombuffer(raw[p:p + stride], dtype=np.uint8).copy()
        p += stride
        if f == 1:  # Sub：同相位字节做累加
            line = (line.reshape(-1, bpp).astype(np.uint32).cumsum(axis=0) & 0xFF).astype(np.uint8).reshape(-1)
        elif f == 2:  # Up
            line = ((line.astype(np.uint16) + prev.astype(np.uint16)) & 0xFF).astype(np.uint8)
        elif f == 3:  # Average
            l16 = line.astype(np.uint16)
            p16 = prev.astype(np.uint16)
            for i in range(bpp, stride):
                l16[i] = (l16[i] + ((l16[i - bpp] + p16[i]) >> 1)) & 0xFF
            line = l16.astype(np.uint8)
        elif f == 4:  # Paeth
            l16 = line.astype(np.int32)
            p16 = prev.astype(np.int32)
            for i in range(stride):
                a = l16[i - bpp] if i >= bpp else 0
                b = int(p16[i])
                c = int(p16[i - bpp]) if i >= bpp else 0
                pp = a + b - c
                pa, pb, pc = abs(pp - a), abs(pp - b), abs(pp - c)
                pred = a if (pa <= pb and pa <= pc) else (b if pb <= pc else c)
                l16[i] = (l16[i] + pred) & 0xFF
            line = l16.astype(np.uint8)
        elif f != 0:
            raise RuntimeError(f"未知的 PNG filter 类型 {f}")
        out[y] = line
        prev = line
    return out


def _decode_png16(path):
    """16-bit RGB/RGBA PNG 解码。PIL 读 16-bit RGB 会截断到高 8 位，为保住精度用 stdlib 自解码。"""
    with open(path, "rb") as fp:
        data = fp.read()
    pos = 8
    idat = bytearray()
    width = height = bit_depth = color_type = interlace = None
    while pos < len(data):
        (length,) = struct.unpack(">I", data[pos:pos + 4])
        pos += 4
        tag = data[pos:pos + 4]
        pos += 4
        payload = data[pos:pos + length]
        pos += length + 4  # 跳过 CRC
        if tag == b"IHDR":
            width, height, bit_depth, color_type, _, _, interlace = struct.unpack(">IIBBBBB", payload)
        elif tag == b"IDAT":
            idat += payload
        elif tag == b"IEND":
            break
    channels = {2: 3, 6: 4}.get(color_type)
    if interlace != 0:
        raise RuntimeError("暂不支持隔行扫描的 PNG，请用 ImageMagick 等工具转成非隔行后再试")
    if channels is None or bit_depth != 16:
        raise RuntimeError(f"不支持的 PNG 格式（color_type={color_type}, bit_depth={bit_depth}），"
                           "HaldCLUT 需要 8/16-bit RGB/RGBA")
    raw = zlib.decompress(bytes(idat))
    bpp = channels * 2
    px = _unfilter_png(raw, height, width * bpp, bpp)
    arr = px.view(np.dtype(">u2")).reshape(height, width, channels)[:, :, :3]
    return arr.astype(np.float32) / 65535.0


def parse_haldclut(path):
    """解析 HaldCLUT PNG：level N 的图尺寸 (N^3)x(N^3)，立方体边长 S=N^2。
    像素按行优先排列，线性序号 i=(b*S+g)*S+r（红最快、蓝最慢，与 .cube 数据序一致），
    返回与 parse_cube 相同的结构，供 apply_cube 三线性插值复用。支持 8/16-bit RGB/RGBA。"""
    return _cached_parse(path, _parse_haldclut_file)


def _parse_haldclut_file(path):
    if not os.path.isfile(path):
        raise RuntimeError(f"找不到 LUT 文件：{path}")
    with open(path, "rb") as fp:
        head = fp.read(26)
    if head[:8] != b"\x89PNG\r\n\x1a\n" or head[12:16] != b"IHDR":
        raise RuntimeError(f"不是有效的 PNG 文件：{path}")
    bit_depth, color_type = head[24], head[25]
    if bit_depth == 16 and color_type in (2, 6):
        rgb = _decode_png16(path)
    else:
        rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    h, w = rgb.shape[:2]
    if h != w:
        raise RuntimeError(f"HaldCLUT 必须是正方形图片，实际 {w}x{h}：{path}")
    level = round(w ** (1.0 / 3.0))
    if level < 2 or level ** 3 != w:
        raise RuntimeError(f"HaldCLUT 边长 {w} 不是合法的 level 立方（8 级=512、12 级=1728、16 级=4096）：{path}")
    s = level * level
    table = torch.from_numpy(np.ascontiguousarray(rgb.reshape(-1, 3))).view(s, s, s, 3)
    return {"type": "3d", "size": s, "table": table,
            "domain_min": [0.0, 0.0, 0.0], "domain_max": [1.0, 1.0, 1.0]}


def apply_cube(image, lut, strength=1.0):
    image = image.float()
    rgb, alpha = split_alpha(image)
    src = rgb.clamp(0.0, 1.0)
    dmin = torch.tensor(lut["domain_min"], device=src.device, dtype=src.dtype).view(1, 1, 1, 3)
    dmax = torch.tensor(lut["domain_max"], device=src.device, dtype=src.dtype).view(1, 1, 1, 3)
    v = ((src - dmin) / (dmax - dmin).clamp_min(1e-6)).clamp(0.0, 1.0)
    table = lut["table"].to(device=src.device, dtype=src.dtype)
    if lut["type"] == "3d":
        b, h, w, _ = src.shape
        t5 = table.permute(3, 0, 1, 2).unsqueeze(0)  # (1,3,S,S,S)：D=b, H=g, W=r
        grid = (v * 2.0 - 1.0).reshape(b, h * w, 1, 1, 3)  # 最后一维 (x=r, y=g, z=b)
        out = F.grid_sample(t5.expand(b, -1, -1, -1, -1), grid, mode="bilinear",
                            padding_mode="border", align_corners=True)
        out = out.reshape(b, 3, h, w).permute(0, 2, 3, 1)
    else:
        s = lut["size"]
        pos = (v * (s - 1)).clamp(0.0, float(s - 1))
        i0 = pos.floor().long().clamp(0, s - 2)
        frac = pos - i0.float()
        chan = torch.arange(3, device=src.device).view(1, 1, 1, 3)
        flat = table.reshape(-1)
        idx = i0 * 3 + chan
        out = flat[idx] + (flat[idx + 3] - flat[idx]) * frac
    out = src + (out.clamp(0.0, 1.0) - src) * float(strength)
    return _join_alpha(out.clamp(0.0, 1.0), alpha)
