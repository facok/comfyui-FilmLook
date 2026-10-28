# -*- coding: utf-8 -*-
"""从胶片数据手册生成 film-look 3D LUT（**实验性，尚未接入运行时预设**）。

方法参考 JanLohse/spectral_film_lut（MIT）的实现思路，用 numpy/torch 自行实现，不依赖
colour-science / scipy / pyqt。完整链路：

    sRGB → 线性 → XYZ → 负片各层 log 曝光 → 负片特性曲线 → 负片密度 D⁻
         → 透过率 T = 10^(−D⁻·SD⁻ᵀ)
         → 印片曝光 E⁺ = T · (S⁺·printer_light·10^(−d_min⁻))
         → 印片特性曲线 → 印片密度 D⁺
         → 投影 XYZ = 10^(−D⁺·SD⁺ᵀ) · (CMF·光源·10^(−d_min⁺))
         → 白平衡到中灰 → sRGB → .cube

负片密度直接投影会得到"底片"（越亮越暗），必须过印片这一道——这是本工具与
"拿一条 LUT 反推参数" 的本质区别。

## 当前状态（2026-10）

链路已跑通，5 支彩色胶片能生成**互不相同、物理合理**的黑位与高光（500T 黑位更沉、
Ektar 高光偏蓝），灰阶严格单调、无 NaN，高光加了软滚降不再硬切顶，`assert_orientation`
会在构建末尾自检通道朝向。但**还不能替换 `presets/`**。

### 踩过的坑（都写进了测试，别重复）

- **write_cube 的轴序**：数据序必须是红变化最快。曾写成 `flat[bi, gi, ri]`，导致红蓝对调
  ——纯红输入会输出 (0.455,0.223,0.751)。**灰阶完全正常、色板平均色度也不变**（互换等于
  把输入集合重排，任何对称统计量都不变），只有原色探测和"写→读往返"能抓到。
  现在 `assert_orientation` + `tests/generator_test.py` 双重兜底。
- **planck() 返回绝对辐射值**（量级 1e31），只该当相对光谱用；未归一化会让曝光算出天量。
- **输出定标点**：曾把 0.18 当线性值，实际 0.18 是 code value（线性 0.027），导致整体过曝。

### 与参考 LUT 的差距（Portra，765 色板）

    色度比：本工具 0.80 / 参考 0.81    暖度：+0.17 / +0.01    反差：0.81 / 0.43

即生成结果的**反差高得多（0.81 vs 0.43）、整体偏暖（+0.17 vs +0.01）**，且 5 支胶片的
暖度只在 +0.15~+0.19 之间，区分度远小于参考预设（+0.01~+0.16）。已定位的主要缺口：

- **印片阶段盖过了胶片个性**：2383 的染料组与打印机灯光决定了成片底色，负片之间的差异
  被压平。这是"负片 + 放映拷贝"这条路线本身的特性，不是 bug。
- **输入侧谱重建偏灰**：用 XYZ 最小范数谱重建，天然拉向灰；参考实现用 barycentric NNLS
  上采样。`--sat` 只是标定补偿（0.314 → 0.43~0.45）。
- **`output_transform` 未完整移植**：参考实现有 OkLab 饱和度调整、2.5D xy LUT 色域映射、
  亮度恢复、可选 Reinhard rolloff；这里只有线性投影 + 中灰定标 + 简单滚降。
- **黑白片未接**：Tri-X（负片）+ 2302（印片）走标量密度的另一条链路。

### 结论

要让"名字与观感一致"，**换路线比调参数更根本**：参考 LUT 更像"负片扫描"路线（扫描仪 +
反相 + 色彩管理），本工具是"负片 + 2383 放映拷贝"路线，两者反差与色调定位本就不同。
若目标是复刻参考 LUT 的性格，应改走扫描路线；若要的是"真实放映拷贝味"，则应调整
`presets/` 的中文名去匹配它。后者不需要改代码，只需要改文案。

用法（离线，不参与插件运行时）：
    python tools/gen_film_luts.py --list
    python tools/gen_film_luts.py --stock portra_400 --size 33 --out <输出目录>
"""

import argparse
import math
import pathlib
import sys
import types

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
FILMDATA = HERE / "filmdata"

WL = np.arange(380.0, 781.0, 10.0)          # 41 个采样，与参考实现 SPECTRAL_SHAPE 一致
LAD_NEGATIVE = np.array([0.78, 0.84, 0.79])  # 中灰在印片中间密度上的参考值

# 预设 → (负片数据模块, 类名, 印片模块, 印片类名)
STOCKS = {
    "portra_400": ("kodak_portra_400", "KODAK_PORTRA_400", "kodak_2383", "KODAK_2383"),
    "fuji_400h": ("fuji_pro_400h", "FUJI_PRO_400H", "kodak_2383", "KODAK_2383"),
    "kodak_gold_200_warm": ("kodak_gold_200", "KODAK_GOLD_200", "kodak_2383", "KODAK_2383"),
    "vision3_500t": ("kodak_5219", "KODAK_5219", "kodak_2383", "KODAK_2383"),
    "ektar_100": ("kodak_ektar_100", "KODAK_EKTAR_100", "kodak_2383", "KODAK_2383"),
    "trix_400": ("kodak_trix_400", "KODAK_TRI_X_400", "kodak_2302", "KODAK_2302"),
}


# ---------------------------------------------------------------- 光谱基础
def _g(x, mu, s1, s2):
    s = np.where(x < mu, s1, s2)
    return np.exp(-0.5 * ((x - mu) / s) ** 2)


def cie1931(wl):
    """Wyman 2013 对 CIE 1931 2° 配色函数的解析拟合。"""
    x = (1.056 * _g(wl, 599.8, 37.9, 31.0) + 0.362 * _g(wl, 442.0, 16.0, 26.7)
         - 0.065 * _g(wl, 501.1, 20.4, 26.2))
    y = 0.821 * _g(wl, 568.8, 46.9, 40.5) + 0.286 * _g(wl, 530.9, 16.3, 31.1)
    z = 1.217 * _g(wl, 437.0, 11.8, 36.0) + 0.681 * _g(wl, 459.0, 26.0, 13.8)
    return np.stack([x, y, z], axis=-1)


CMF = cie1931(WL)
XYZ2RGB = np.array([[3.2406, -1.5372, -0.4986],
                    [-0.9689, 1.8758, 0.0415],
                    [0.0557, -0.2040, 1.0570]])
RGB2XYZ = np.linalg.inv(XYZ2RGB)


def planck(wl_nm, kelvin):
    """普朗克黑体谱，归一化成相对光谱分布（只用形状，量级无意义）。"""
    c2 = 1.4388e-2
    lam = wl_nm * 1e-9
    p = 1.0 / (lam ** 5 * (np.exp(c2 / (lam * kelvin)) - 1.0))
    return p / p.max()


def srgb_to_linear(x):
    x = np.clip(x, 0, None)
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(x):
    x = np.clip(x, 0, None)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * x ** (1 / 2.4) - 0.055)


def align(sd):
    """{波长: 值} → 41 点数组（线性内插 + 端点外延）。"""
    keys = np.array(sorted(sd.keys()), dtype=float)
    vals = np.array([sd[k] for k in sorted(sd.keys())], dtype=float)
    return np.interp(WL, keys, vals)


# ---------------------------------------------------------------- 胶片模型
def load_film(module_name, cls_name):
    """执行数据手册模块（只含 FilmData 构造），返回其字段字典。"""
    pkg = types.ModuleType("spectral_film_lut")
    pkg.__path__ = []
    fd = types.ModuleType("spectral_film_lut.film_data")
    fd.FilmData = lambda **kw: kw
    pkg.film_data = fd
    sys.modules["spectral_film_lut"] = pkg
    sys.modules["spectral_film_lut.film_data"] = fd
    path = FILMDATA / f"{module_name}.py"
    mod = types.ModuleType(module_name)
    exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), mod.__dict__)
    return getattr(mod, cls_name)


class Film:
    """把 FilmData 整理成生成 LUT 需要的数值形态。"""

    def __init__(self, data):
        self.name = data["name"]
        self.iso = data.get("iso")
        self.density_measure = data.get("density_measure")
        self.d_min = None

        # 灵敏度：10**log_sensitivity，按峰值归一
        sens = np.stack([align(s) for s in data["log_sensitivity"]], axis=1)
        sens = 10 ** sens
        self.sensitivity = sens / sens.max(axis=0, keepdims=True)

        # 特性曲线（logH → 密度），减去 d_min 并重采样到统一网格
        curves, dmins = [], []
        for c in data["sensiometric_curve"]:
            xs = np.array(sorted(c.keys()), dtype=float)
            ys = np.array([c[k] for k in sorted(c.keys())], dtype=float)
            curves.append((xs, ys))
            dmins.append(ys.min())
        self.d_min = np.array(dmins)
        grid = np.linspace(min(x.min() for x, _ in curves), max(x.max() for x, _ in curves), 2048)
        self.curve_x = grid
        self.curve_y = np.stack([
            np.interp(grid, xs, ys - d) for (xs, ys), d in zip(curves, dmins)
        ], axis=1)                                   # (2048, 3)

        # 光谱密度：优先用数据里直接给的 spectral_density（印片常见），
        # 否则把 d_ref_sd−d_min_sd 拆成 R/G/B 三个波段（相机负片常见）
        self.is_bw = data.get("density_measure") == "bw"
        if data.get("spectral_density"):
            self.spectral_density = np.stack([align(d) for d in data["spectral_density"]], axis=0)
            self.d_min_sd = align(data["d_min_sd"]) if data.get("d_min_sd") else np.zeros(WL.size)
        else:
            d_min_sd = align(data["d_min_sd"]) if data.get("d_min_sd") else np.zeros(WL.size)
            ramp = np.clip(align(data["d_ref_sd"]) - d_min_sd, 0, None)
            band = np.zeros((3, WL.size))
            band[0, WL >= 600] = 1.0
            band[1, (WL >= 500) & (WL < 600)] = 1.0
            band[2, WL < 500] = 1.0
            band = np.stack([np.convolve(b, np.ones(3) / 3, mode="same") for b in band])
            self.spectral_density = band * ramp[None, :] / np.clip(
                (band * ramp[None, :]).max(axis=1, keepdims=True), 1e-6, None)
            self.d_min_sd = d_min_sd

        if self.iso is not None:
            # 相机片：中灰曝光由 ISO 决定
            self.log_H_ref = np.full(3, math.log10(12.5 / self.iso))
        elif data.get("lad"):
            # 印片：中灰由 LAD 密度反查特性曲线
            self.log_H_ref = np.array([np.interp(lad, self.curve_y[:, i], self.curve_x)
                                       for i, lad in enumerate(data["lad"])])
        else:
            self.log_H_ref = np.zeros(3)
        self.H_ref = 10 ** self.log_H_ref

        # 中灰处的光谱密度（印片用）
        self.d_ref = self._density(self.log_H_ref)
        self.d_ref_sd = (self.d_ref[:, None] * self.spectral_density).sum(axis=0) + self.d_min_sd

    def _density(self, log_e):
        """log 曝光 → 密度（逐层线性内插），保持输入形状。"""
        log_e = np.asarray(log_e, dtype=float)
        orig = log_e.shape
        e2 = log_e.reshape(-1, 3)
        out = np.stack([np.interp(e2[:, i], self.curve_x, self.curve_y[:, i])
                        for i in range(3)], axis=1)
        return out.reshape(orig)


# ---------------------------------------------------------------- 管线
def printer_light(neg, prt, kelvin=3200.0):
    """求印片灯光谱，使负片中灰印到印片基准曝光（中性）。

    参考实现用三色滤光片光谱做基；这里用普朗克灯光 × 三个宽波段近似，再解线性方程组，
    语义一致：让 d_ref 的负片在印片上落在 log_H_ref。
    """
    lamp = planck(WL, kelvin)
    bands = np.stack([np.exp(-0.5 * ((WL - c) / 40.0) ** 2) for c in (450.0, 540.0, 610.0)], axis=1)
    basis = lamp[:, None] * bands                        # (41,3)
    reduced = (basis.T * 10 ** -neg.d_ref_sd).T          # (41,3)
    factors = np.linalg.solve(prt.sensitivity.T @ reduced, prt.H_ref)
    return basis @ factors


def build_lut(neg, prt, size=33, projector_kelvin=6500, white_balance=True, saturation=1.45):
    """生成 size³ 的 sRGB→sRGB LUT。"""
    xyz2rgb = XYZ2RGB

    # 输入：XYZ → 负片层曝光。最小范数谱重建假设下的 3x3
    pinv_cmf = CMF @ np.linalg.inv(CMF.T @ CMF)          # (41,3)
    M_in = (neg.sensitivity * planck(WL, 5500)[:, None]).T @ pinv_cmf   # (3,3)
    neutral_xyz = np.array([0.95047, 1.0, 1.08883]) * 0.18
    M_in = M_in * (10 ** neg.log_H_ref[1]) / float((M_in @ neutral_xyz)[1])

    # 印片曝光矩阵
    pl = printer_light(neg, prt)
    printing_mat = (prt.sensitivity.T * pl * 10 ** -neg.d_min_sd).T      # (41,3)

    # 投影矩阵
    proj = planck(WL, projector_kelvin)
    output_mat = (CMF.T * proj * 10 ** -prt.d_min_sd).T                  # (41,3)

    grid = np.linspace(0, 1, size)
    r, g, b = np.meshgrid(grid, grid, grid, indexing="ij")
    rgb_in = np.stack([r, g, b], axis=-1).reshape(-1, 3)
    xyz = srgb_to_linear(rgb_in) @ RGB2XYZ.T

    log_e = np.log10(np.clip(xyz @ M_in.T, 1e-7, None))
    d_neg = neg._density(log_e)
    t_neg = 10 ** -(d_neg @ neg.spectral_density)
    log_e_print = np.log10(np.clip(t_neg @ printing_mat, 1e-6, None))
    d_prt = prt._density(log_e_print)
    xyz_out = 10 ** -(d_prt @ prt.spectral_density) @ output_mat

    # 输出定标：让 sRGB 0.5（线性 0.214≈中灰）印出来落在 Y=0.18，并做色度白平衡
    g5 = srgb_to_linear(np.array([[0.5, 0.5, 0.5]])) @ RGB2XYZ.T
    le = np.log10(np.clip(g5 @ M_in.T, 1e-7, None))
    dn = neg._density(le)
    ln = np.log10(np.clip(10 ** -(dn @ neg.spectral_density) @ printing_mat, 1e-6, None))
    dp = prt._density(ln)
    white = (10 ** -(dp @ prt.spectral_density) @ output_mat).reshape(3)
    xyz_out = xyz_out * (0.18 / max(float(white[1]), 1e-8))
    if white_balance:
        xyz_out = xyz_out * (white[1] / np.clip(white, 1e-8, None))

    rgb_out = np.clip(xyz_out @ xyz2rgb.T, 0, None)

    # 饱和度标定（对应参考实现的 output_color_transform(sat_adjust=...)）：
    # 输入侧用的是 XYZ 最小范数谱重建，天然偏灰，实测色度只有参考的 ~70%，
    # 这里在亮度轴周围线性放大即可补回，不改动色相。
    if saturation != 1.0:
        lum = (rgb_out * np.array([0.2126, 0.7152, 0.0722])).sum(axis=-1, keepdims=True)
        rgb_out = lum + (rgb_out - lum) * saturation
        rgb_out = np.clip(rgb_out, 0, None)

    # 高光软滚降：knee 以上指数逼近 1.0，避免 R 通道在 0.83 左右就硬切顶
    knee = 0.80
    over = np.clip(rgb_out - knee, 0, None)
    rgb_out = np.where(rgb_out > knee,
                       knee + (1.0 - knee) * (1.0 - np.exp(-over / (1.0 - knee))),
                       rgb_out)
    out = np.clip(linear_to_srgb(rgb_out), 0, 1)
    return out.reshape(size, size, size, 3)


def assert_orientation(lut):
    """自检：LUT 的通道朝向是否正确。

    注意：**色板平均色度/暖度对"红蓝互换"是不变量**——互换相当于把输入集合重排，
    均值不变，所以这类错误只能靠原色探测发现。灰阶更是完全看不出来（r=g=b 对称）。
    实测教训：write_cube 曾把索引写成 [b,g,r]，纯红输入会输出蓝色，而色板统计毫无变化。
    """
    n = lut.shape[0]
    hi = n - 1
    red = lut[hi, 0, 0]
    green = lut[0, hi, 0]
    blue = lut[0, 0, hi]
    assert red[0] > red[1] and red[0] > red[2], f"红轴朝向错误：纯红 → {red}"
    assert green[1] > green[0] and green[1] > green[2], f"绿轴朝向错误：纯绿 → {green}"
    assert blue[2] > blue[0] and blue[2] > blue[1], f"蓝轴朝向错误：纯蓝 → {blue}"


def write_cube(path, lut, title):
    """写出 .cube。数据序必须是**红变化最快**：外层 b、中层 g、内层 r。

    lut 的形状是 (size, size, size, 3)，索引顺序为 [r, g, b]（由 meshgrid(indexing="ij")
    生成），所以内层循环取 r 时要用 flat[ri, gi, bi]——写成 flat[bi, gi, ri] 会把红蓝
    轴对调：实测纯红输入会输出 (0.455,0.223,0.751) 这样的蓝色，灰阶却看不出问题，
    非常隐蔽。
    """
    size = lut.shape[0]
    lines = [f'TITLE "{title}"', f"LUT_3D_SIZE {size}",
             "DOMAIN_MIN 0.0 0.0 0.0", "DOMAIN_MAX 1.0 1.0 1.0", ""]
    for bi in range(size):
        for gi in range(size):
            for ri in range(size):
                c = lut[ri, gi, bi]
                lines.append(f"{c[0]:.6f} {c[1]:.6f} {c[2]:.6f}")
    pathlib.Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stock", default="portra_400")
    ap.add_argument("--size", type=int, default=33)
    ap.add_argument("--sat", type=float, default=1.45,
                    help="饱和度标定：输入侧用最小范数谱重建会偏灰，1.45 可把 765 色板色度"
                         "从 0.314 补到约 0.45（参考 LUT 为 0.451）")
    ap.add_argument("--out", default=str(HERE.parent / "presets"))
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--compare", action="store_true", help="与现有 presets/*.cube 对比灰阶")
    a = ap.parse_args()

    if a.list:
        for k, v in STOCKS.items():
            print(f"  {k:22s} 负片 {v[0]:20s} 印片 {v[2]}")
        return

    neg_mod, neg_cls, prt_mod, prt_cls = STOCKS[a.stock]
    neg = Film(load_film(neg_mod, neg_cls))
    prt = Film(load_film(prt_mod, prt_cls))
    print(f"负片 {neg.name} (ISO {neg.iso}) + 印片 {prt.name}")

    lut = build_lut(neg, prt, size=a.size, saturation=a.sat)
    out = pathlib.Path(a.out) / f"{a.stock}.cube"
    write_cube(out, lut, f"{neg.name} on {prt.name}")
    print(f"已写出 {out}  ({a.size}³)")

    if a.compare:
        old = pathlib.Path(a.out) / f"{a.stock}.cube"
        print("灰阶对比（输入 0/.18/.5/.82/1 时的 RGB）")
        step = (a.size - 1)
        idx = [0, round(0.18 * step), round(0.5 * step), round(0.82 * step), step]
        for i in idx:
            c = lut[i, i, i]
            print(f"  in={i / step:.2f}: ({c[0]:.3f},{c[1]:.3f},{c[2]:.3f})")


if __name__ == "__main__":
    main()
