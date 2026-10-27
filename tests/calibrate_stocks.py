# -*- coding: utf-8 -*-
"""用参考 .cube LUT 标定内置胶片预设：Adam 可微拟合 + 灰卡/彩色块探针 + 性格可见度验收。
用法：python tests/calibrate_stocks.py [步数] [substitute|true|true2|merged] [预设名过滤]
曲线参数为共享标量（有界）+ 每通道 ±0.1 tanh 偏移，防止逐通道打架导致预设退化。"""

import importlib.util
import pathlib
import sys

import torch
import torch.nn.functional as F

PKG_DIR = pathlib.Path(__file__).resolve().parent.parent

spec = importlib.util.spec_from_file_location(
    "comfyui_FilmLook", PKG_DIR / "__init__.py", submodule_search_locations=[str(PKG_DIR)])
mod = importlib.util.module_from_spec(spec)
sys.modules["comfyui_FilmLook"] = mod
spec.loader.exec_module(mod)

film_core = mod.film_core
FILM_STOCKS = mod.film_stocks.FILM_STOCKS

# 替代参考（MIT: poptechstudio/ai-cinema-studio-engine，光谱数据推导）
STOCK_TO_LUT_SUBSTITUTE = {
    "清新人像 (Portra 400)": "fuji_eterna_250d.cube",
    "日系空气感 (Fuji 400H)": "faded_film_aged.cube",
    "暖阳金调 (Kodak Gold)": "16mm_reversal_vintage.cube",
    "冷调电影感 (Vision3 500T)": "kodak_vision3_500t.cube",
    "浓郁鲜艳 (Ektar)": "kodak_ektachrome_100d.cube",
    "黑白纪实 (Tri-X)": "kodak_trix_400_bw.cube",
}

# 真身参考（用户 luts/ 收藏中的同名胶片 LUT，更准确）
STOCK_TO_LUT_TRUE = {
    "清新人像 (Portra 400)": "Neg_Kodak_Portra_160_VC.cube",
    "浓郁鲜艳 (Ektar)": "Neg_Kodak_Ektar_100.cube",
    "黑白纪实 (Tri-X)": "BW_Kodak_TriX_400.cube",
}

# 真身参考 v2：YahiaAngelo/Film-Luts（G'MIC/Pat David 胶片集，CC BY-SA 数据，仅标定用）
# 与 dazz-retro-camera（MIT）的 Kodak Gold 200 Warm；位于 tests/reference_luts/
STOCK_TO_LUT_TRUE2 = {
    "日系空气感 (Fuji 400H)": "tests/reference_luts/fuji_400h.cube",
    "清新人像 (Portra 400)": "tests/reference_luts/kodak_portra_400.cube",
    "暖阳金调 (Kodak Gold)": "tests/reference_luts/cpm35_kodak_gold_warm.cube",
}

# 默认：真身优先的合并映射；CLI 第二参数可选 substitute / true / true2 只跑某一组
STOCK_TO_LUT = {**STOCK_TO_LUT_SUBSTITUTE, **STOCK_TO_LUT_TRUE, **STOCK_TO_LUT_TRUE2}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
GRAY_ANCHOR_W = 20.0   # 灰轴亮度锚定权重
GRAY_CHROMA_W = 45.0   # 灰轴色偏(R-G/G-B)对齐权重（曲线已有结构约束压不塌，可放心加回）
MONO_NEUTRAL_W = 5.0   # 黑白预设灰轴中性约束
CURVE_REG_W = 0.005    # 反退化：拉 gamma→1.0、toe/shoulder→0.5，防止曲线塌平
XT_REG_W = 1e-4        # 串扰偏离单位阵的微小惩罚，防过拟合

# 曲线结构约束：共享标量（有界）+ 每通道 ±0.1 tanh 偏移——杜绝逐通道互相打架
GAMMA_RANGE = (0.75, 1.35)
TOE_RANGE = (0.1, 1.5)
SHOULDER_RANGE = (0.1, 1.5)
CHANNEL_OFFSET = 0.1


def make_palette():
    lv = torch.linspace(0.0, 1.0, 9)
    r, g, b = torch.meshgrid(lv, lv, lv, indexing="ij")
    grid = torch.stack([r, g, b], dim=-1).reshape(-1, 3)          # 729 色网格
    grays = torch.linspace(0.0, 1.0, 24).unsqueeze(-1).expand(-1, 3)  # 24 级灰阶
    extras = torch.tensor([
        (0.87, 0.66, 0.55), (0.72, 0.55, 0.44), (0.55, 0.38, 0.30),  # 肤色
        (0.45, 0.65, 0.90), (0.30, 0.50, 0.85), (0.60, 0.75, 0.95),  # 天空
        (0.25, 0.55, 0.20), (0.15, 0.40, 0.15), (0.55, 0.65, 0.25),  # 植物
        (0.60, 0.20, 0.15), (0.80, 0.60, 0.20), (0.20, 0.35, 0.60),  # 红砖/黄/牛仔
    ])
    pal = torch.cat([grid, grays, extras], dim=0)
    # 规避精确 0：特性曲线端点归一化后 y=0，y**k 对指数求导会产生 NaN 梯度
    return pal.clamp_min(1e-3).reshape(1, -1, 1, 3)


def _inv_sigmoid(v):
    v = v.clamp(1e-4, 1.0 - 1e-4)
    return torch.log(v / (1.0 - v))


def fit_stock(label, ref_lut, steps, lr=0.005):
    stock0 = FILM_STOCKS[label]
    mono = bool(stock0.get("monochrome"))
    pal = make_palette().to(DEVICE)
    ref = film_core.apply_cube(pal, ref_lut, 1.0)
    gray = torch.linspace(1e-3, 1.0, 32).view(1, 32, 1, 1).expand(1, 32, 1, 3).contiguous().to(DEVICE)
    gray_ref = film_core.apply_cube(gray, ref_lut, 1.0)
    T = lambda seq: torch.tensor(seq, device=DEVICE, dtype=torch.float32)
    # sigmoid 反值钳到 ±4 以内：初值过小（如 black_lift=0）会饱和导致梯度消失、参数卡死
    isg = lambda v: _inv_sigmoid(v).clamp(-4.0, 4.0)
    isg_amt = lambda v: _inv_sigmoid(v / 0.35).clamp(-4.0, 4.0)  # 分离色调 amt 上限 0.35
    def range_param(mean3, lo, hi):
        # 共享标量 init 用当前三通道均值（钳进边界），通道偏移从 0 起步——不继承上轮逐通道打架的值
        v = min(max(float(sum(mean3)) / 3.0, lo + 1e-3), hi - 1e-3)
        return torch.nn.Parameter(_inv_sigmoid(T([(v - lo) / (hi - lo)])).clamp(-4.0, 4.0))

    c0, st0 = stock0["curve"], stock0["split_tone"]
    params = {
        "gamma": range_param(c0["gamma"], *GAMMA_RANGE),
        "gamma_off": torch.nn.Parameter(torch.zeros(3, device=DEVICE)),
        "toe": range_param(c0["toe"], *TOE_RANGE),
        "toe_off": torch.nn.Parameter(torch.zeros(3, device=DEVICE)),
        "shoulder": range_param(c0["shoulder"], *SHOULDER_RANGE),
        "shoulder_off": torch.nn.Parameter(torch.zeros(3, device=DEVICE)),
        "shadow_col": torch.nn.Parameter(isg(T(st0["shadow"]))),
        "shadow_amt": torch.nn.Parameter(isg_amt(T([st0["shadow_amt"]]))),
        "highlight_col": torch.nn.Parameter(isg(T(st0["highlight"]))),
        "highlight_amt": torch.nn.Parameter(isg_amt(T([st0["highlight_amt"]]))),
        "black_lift": torch.nn.Parameter(isg(T(stock0["black_lift"]) / 0.05)),
    }
    if not mono:
        params["sat_scale"] = torch.nn.Parameter(isg(T([stock0["sat_scale"]]) - 0.5))
    # 黑白预设也拟合串扰：mono 前的 3x3 矩阵等价于自由的三通道亮度权重，
    # 可对齐黑白胶片的光谱感光特性（与 Rec.709 不同），且不影响灰轴中性
    params["crosstalk"] = torch.nn.Parameter(T(stock0["crosstalk"]))

    def shared(name, lo, hi):
        return lo + (hi - lo) * torch.sigmoid(params[name])

    def build():
        gamma = shared("gamma", *GAMMA_RANGE) + CHANNEL_OFFSET * torch.tanh(params["gamma_off"])
        toe = shared("toe", *TOE_RANGE) + CHANNEL_OFFSET * torch.tanh(params["toe_off"])
        shoulder = shared("shoulder", *SHOULDER_RANGE) + CHANNEL_OFFSET * torch.tanh(params["shoulder_off"])
        return {
            "curve": {"gamma": gamma, "toe": toe, "shoulder": shoulder},
            "crosstalk": params["crosstalk"] if "crosstalk" in params else stock0["crosstalk"],
            "split_tone": {"shadow": torch.sigmoid(params["shadow_col"]),
                           "shadow_amt": 0.35 * torch.sigmoid(params["shadow_amt"]),
                           "highlight": torch.sigmoid(params["highlight_col"]),
                           "highlight_amt": 0.35 * torch.sigmoid(params["highlight_amt"])},
            "sat_scale": (0.5 + torch.sigmoid(params["sat_scale"])) if "sat_scale" in params else stock0["sat_scale"],
            "black_lift": 0.05 * torch.sigmoid(params["black_lift"]),
            "monochrome": mono,
        }

    def rmse(stock):
        out = film_core.apply_film_color(pal, stock, strength=1.0, auto_exposure=False)
        return float(torch.sqrt(F.mse_loss(out, ref)))

    rmse_before = rmse(stock0)
    eye = torch.eye(3, device=DEVICE).reshape(-1)
    opt = torch.optim.Adam(params.values(), lr=lr)
    for i in range(steps):
        opt.zero_grad()
        out = film_core.apply_film_color(pal, build(), strength=1.0, auto_exposure=False)
        loss = F.mse_loss(out, ref)
        # 灰轴锚定：729 网格中饱和色占绝对多数，单独加重灰阶避免灰轴被全局 MSE 牺牲
        gout = film_core.apply_film_color(gray, build(), strength=1.0, auto_exposure=False)
        loss = loss + GRAY_ANCHOR_W * F.mse_loss(gout, gray_ref)
        # 灰轴色偏对齐（适中权重）：灰轴 R-G/G-B 偏差贴参考，但不许以压塌曲线为代价
        loss = loss + GRAY_CHROMA_W * (F.mse_loss(gout[..., 0] - gout[..., 1], gray_ref[..., 0] - gray_ref[..., 1])
                                       + F.mse_loss(gout[..., 1] - gout[..., 2], gray_ref[..., 1] - gray_ref[..., 2]))
        # 反退化：曲线共享参数不许塌成弱曲线
        loss = loss + CURVE_REG_W * ((shared("gamma", *GAMMA_RANGE) - 1.0) ** 2
                                     + (shared("toe", *TOE_RANGE) - 0.5) ** 2
                                     + (shared("shoulder", *SHOULDER_RANGE) - 0.5) ** 2)
        if not mono:
            loss = loss + XT_REG_W * ((params["crosstalk"] - eye) ** 2).sum()
        if mono:
            # 黑白预设加灰轴中性约束：参考 LUT 输出严格 R=G=B，防止分离色调把灰阶带偏色
            loss = loss + MONO_NEUTRAL_W * ((out[..., 0] - out[..., 1]) ** 2 + (out[..., 1] - out[..., 2]) ** 2).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(params.values()), 1.0)
        opt.step()
    fitted = _plain(build())
    rmse_after = rmse(fitted)

    fitted_vals = {
        "gamma": _r3(fitted["curve"]["gamma"]), "toe": _r3(fitted["curve"]["toe"]),
        "shoulder": _r3(fitted["curve"]["shoulder"]),
        "crosstalk": [round(float(x), 4) for x in fitted["crosstalk"]],
        "shadow": _r3(fitted["split_tone"]["shadow"]),
        "shadow_amt": round(fitted["split_tone"]["shadow_amt"], 4),
        "highlight": _r3(fitted["split_tone"]["highlight"]),
        "highlight_amt": round(fitted["split_tone"]["highlight_amt"], 4),
        "sat_scale": round(fitted["sat_scale"], 4),
        "black_lift": _r3(fitted["black_lift"]),
    }
    return stock0, fitted, fitted_vals, rmse_before, rmse_after


def _r3(seq):
    return tuple(round(float(v), 4) for v in seq)


def _plain(stock):
    """把含可微 tensor 的 stock 转成纯数值 dict（探针与生产路径一致，不建梯度图）。"""
    out = {}
    for k, v in stock.items():
        if torch.is_tensor(v):
            flat = v.detach().cpu().reshape(-1)
            out[k] = float(flat[0]) if flat.numel() == 1 else [float(x) for x in flat]
        elif isinstance(v, dict):
            out[k] = _plain(v)
        else:
            out[k] = v
    return out


GRAY_LEVELS = (0.12, 0.46, 0.85)
PATCHES = {"红砖": (0.60, 0.20, 0.15), "草绿": (0.25, 0.55, 0.20), "肤色": (0.72, 0.55, 0.44)}


def _apply(what, pal):
    if isinstance(what, dict) and "table" in what:
        return film_core.apply_cube(pal, what, 1.0)
    return film_core.apply_film_color(pal, what, strength=1.0, auto_exposure=False)


def gray_probe(what):
    pal = torch.tensor([[v, v, v] for v in GRAY_LEVELS]).reshape(1, -1, 1, 3)
    out = _apply(what, pal)[0, :, 0, :]
    return [(round(float(o[0] - o[2]), 4), round(float(o[1] - o[0]), 4)) for o in out]


def gray_ramp_mae(what):
    """灰阶亮度曲线的平均绝对误差 + 黑/白点（与参考 LUT 对照用）。"""
    pal = torch.linspace(0.0, 1.0, 11).view(1, 11, 1, 1).expand(1, 11, 1, 3).contiguous()
    return _apply(what, pal)[0, :, 0, :].mean(dim=-1)


def chroma_probe(what):
    pal = torch.tensor(list(PATCHES.values())).reshape(1, -1, 1, 3)
    out = _apply(what, pal)[0, :, 0, :]
    return [round(float(o.max() - o.min()), 4) for o in out]


def _resolve_lut(lut_file):
    p = PKG_DIR / lut_file
    return str(p if p.is_file() else PKG_DIR / "luts" / lut_file)


def main():
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    which = sys.argv[2] if len(sys.argv) > 2 else "merged"
    filt = sys.argv[3] if len(sys.argv) > 3 else ""
    lr = float(sys.argv[4]) if len(sys.argv) > 4 else 0.005
    mapping = {"substitute": STOCK_TO_LUT_SUBSTITUTE, "true": STOCK_TO_LUT_TRUE,
               "true2": STOCK_TO_LUT_TRUE2}.get(which, STOCK_TO_LUT)
    mapping = {k: v for k, v in mapping.items() if filt in k}
    print(f"device={DEVICE} steps={steps} lr={lr} 参考集={which} 色板颜色数={make_palette().shape[1]}")
    results = {}
    for label, lut_file in mapping.items():
        ref_lut = film_core.parse_cube(_resolve_lut(lut_file))
        stock0, fitted, fitted_vals, rmse_before, rmse_after = fit_stock(label, ref_lut, steps, lr)
        results[label] = (stock0, fitted, fitted_vals, ref_lut, lut_file, rmse_before, rmse_after)
        print(f"{label:22s} <- {lut_file:28s} RMSE 拟合前 {rmse_before:.4f} -> 拟合后 {rmse_after:.4f}")

    print("\n=== 灰卡探针：R-B / G-R（暗部 0.12 / 中间 0.46 / 高光 0.85，sRGB 差值）===")
    for label, (stock0, fitted, _, ref_lut, _, _, _) in results.items():
        print(f"\n[{label}]")
        print(f"  {'灰阶':6s} {'拟合前':>18s} {'拟合后':>18s} {'参考LUT':>18s}")
        ok = True
        for i, v in enumerate(GRAY_LEVELS):
            a, b, c = gray_probe(stock0)[i], gray_probe(fitted)[i], gray_probe(ref_lut)[i]
            print(f"  {v:<6.2f} {str(a):>18s} {str(b):>18s} {str(c):>18s}")
            for j in range(2):  # 灰轴不翻车：|ours| ≤ 1.5x|ref|（参考近零时留 0.005 绝对容差）
                if abs(b[j]) > 1.5 * abs(c[j]) + 0.005:
                    ok = False
        print(f"  灰轴 1.5x 验收: {'PASS' if ok else 'FAIL'}")
        ra = gray_ramp_mae(stock0)
        rb = gray_ramp_mae(fitted)
        rc = gray_ramp_mae(ref_lut)
        print(f"  灰阶亮度 MAE: 拟合前 {float((ra - rc).abs().mean()):.4f} -> 拟合后 {float((rb - rc).abs().mean()):.4f}")
        print(f"  黑点/白点:   拟合后 ({float(rb[0]):.3f}, {float(rb[-1]):.3f})  参考 ({float(rc[0]):.3f}, {float(rc[-1]):.3f})")

    print("\n=== 彩色块色彩度（max-min，越大越饱和；>0.95 视为削波警告）===")
    for label, (stock0, fitted, _, ref_lut, _, _, _) in results.items():
        before, after, ref = chroma_probe(stock0), chroma_probe(fitted), chroma_probe(ref_lut)
        for i, name in enumerate(PATCHES):
            flag = "  <-- 削波!" if after[i] > 0.95 else ""
            print(f"  [{label}] {name}: 拟合前 {before[i]:.4f} -> 拟合后 {after[i]:.4f} (参考 {ref[i]:.4f}){flag}")

    print("\n=== 性格可见度（765 色板，strength=1 无智能曝光）===")
    pal = make_palette()
    color_labels = [l for l in results if not FILM_STOCKS[l]["monochrome"]]
    outs = {label: film_core.apply_film_color(pal, fitted, strength=1.0, auto_exposure=False)
            for label, (_, fitted, _, _, _, _, _) in results.items()}
    for l in color_labels:
        d = float((outs[l] - pal).abs().mean())
        print(f"  [{l}] vs 原图 平均|ΔRGB| = {d:.4f} {'PASS' if d >= 0.030 else 'FAIL <0.030'}")
    print("  预设两两区分度（彩色预设，平均|Δ|，要求 ≥0.025）:")
    worst = (None, 1e9)
    for i, a in enumerate(color_labels):
        for b in color_labels[i + 1:]:
            d = float((outs[a] - outs[b]).abs().mean())
            if d < worst[1]:
                worst = (f"{a.split(' ')[0]} vs {b.split(' ')[0]}", d)
            print(f"    {a.split(' ')[0]} vs {b.split(' ')[0]}: {d:.4f}{'' if d >= 0.025 else '  <-- FAIL'}")
    if worst[0]:
        print(f"  最小区分度: {worst[0]} = {worst[1]:.4f}")

    print("\n=== 拟合参数（用于写回 film_stocks.py）===")
    for label, (_, _, v, _, lut_file, _, _) in results.items():
        xt = ",\n                      ".join(
            ", ".join(f"{x:.4f}" for x in v["crosstalk"][i * 3:(i + 1) * 3]) for i in range(3))
        print(f'    "{label}": {{  # 参考 {lut_file}')
        print(f'        "curve": {{"gamma": {v["gamma"]}, "toe": {v["toe"]}, "shoulder": {v["shoulder"]}}},')
        print(f'        "crosstalk": [{xt}],')
        print(f'        "split_tone": {{"shadow": {v["shadow"]}, "shadow_amt": {v["shadow_amt"]},')
        print(f'                       "highlight": {v["highlight"]}, "highlight_amt": {v["highlight_amt"]}}},')
        print(f'        "sat_scale": {v["sat_scale"]},')
        print(f'        "black_lift": {v["black_lift"]},')
        print(f'        "monochrome": {FILM_STOCKS[label]["monochrome"]},')
        print("    },")


if __name__ == "__main__":
    main()
