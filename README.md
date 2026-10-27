# FilmLook 胶片质感套件

为 ComfyUI 带来一套完整的胶片质感流程：胶片调色、LUT 应用、高光光晕（Halation）、物理胶片颗粒，以及一个把三者串起来的"一键胶片"节点。

设计原则：**技术参数藏进预设，界面只说人话**。曝光、高光检测、肤色保护都默认自动完成，你只需要选风格、拉强度。

## 功能一览

| 节点 | 分类 | 作用 |
|---|---|---|
| FilmLook 胶片调色 | FilmLook/色彩 | 7 种胶片预设 + 亮度/对比度/饱和度/色温/色调手动微调 |
| FilmLook LUT 应用 | FilmLook/色彩 | 应用 `.cube` 或 HaldCLUT `.png` LUT 文件，可自定义路径 |
| FilmLook 光晕 (Halation) | FilmLook/光学 | 高光周围的红橙色雾圈，胶片标志性光学效果 |
| FilmLook 镜头质感 | FilmLook/光学 | 柔光（黑柔镜感）+ 暗角，两个滑杆 |
| FilmLook 胶片颗粒 | FilmLook/颗粒 | 基于 silvergrain 的物理银盐颗粒（非普通噪点） |
| FilmLook 一键胶片 | FilmLook | 调色 + 光晕 + 颗粒一次完成 |

## 安装

1. 把本文件夹整个复制到 `ComfyUI/custom_nodes/` 下（文件夹名保持 `comfyui-FilmLook`）。
2. 安装基础依赖（ComfyUI 一般已自带）：
   ```
   pip install -r requirements.txt
   ```
3. **手动安装颗粒引擎 silvergrain**（只有"胶片颗粒"和"一键胶片"节点需要）：
   ```
   pip install silvergrain[gpu]
   ```
   - 需要 NVIDIA CUDA 显卡才有实用速度：GPU 渲染 1024×1024 约 0.2 秒，CPU 约 159 秒。
   - silvergrain 采用 **AGPLv3** 开源协议，商用前请确认合规。
   - 不安装也不影响其余三个节点。
4. 重启 ComfyUI。

## 节点用法（各一句话）

- **胶片调色**：接入图像 → 选预设 → 调 `strength` 即可，其余参数默认就很稳。
- **LUT 应用**：把 `.cube` 或 HaldCLUT `.png` 文件放进插件的 `luts/` 文件夹后重启，在下拉框中选择；或选"自定义路径"并填完整路径。
- **光晕**：默认自动检测高光，只调 `strength` 和 `radius` 两档就够。
- **胶片颗粒**：选颗粒粗细（ISO）→ 选亮度/彩色颗粒 → 完成。建议 GPU 环境使用。
- **一键胶片**：选预设和颗粒档位即可出片，适合批量出图。

## 参数说明

### FilmLook 胶片调色

| 参数 | 默认 | 说明 |
|---|---|---|
| preset | 清新人像 (Portra 400) | 胶片风格预设，详见下方预设表 |
| strength | 0.85 | 效果强度，0 为原图 |
| auto_exposure | 开 | 智能曝光：自动把整体亮度校正到胶片友好区间 |
| brightness | 0 | 亮度，+1 约提亮一档，-1 约压暗一档 |
| contrast | 1.0 | 对比度，1 保持不变 |
| saturation | 1.0 | 饱和度；提升时优先增强灰暗颜色，肤色更自然 |
| temperature | 0 | 色温：负偏冷（蓝），正偏暖（黄）；满行程约 ±65 mired（等效色温 6500K → 4600K / 11600K） |
| tint | 0 | 色调：负偏绿，正偏品红；与色温同量级，只改色相不改亮度，中灰亮度守恒 |
| preserve_luminance | 0 | 亮度保持：0 允许胶片曲线自由改变亮度，1 输出亮度完全贴回原图；遇到过曝时调高 |
| style_boost | 1.0 | 风格浓度：1 为忠实胶片效果，>1 把风格放大到超出真实胶片 |

### 胶片预设速查

6 个胶片预设已经过参考 LUT 标定：参考数据来自 MIT 授权的 [poptechstudio/ai-cinema-studio-engine](https://github.com/poptechstudio/ai-cinema-studio-engine)（其 LUT 由柯达/富士数据手册的光谱特性曲线数学推导）、G'MIC/Pat David 胶片集（经 YahiaAngelo/Film-Luts 分发，CC BY-SA 数据，仅作标定参考）与 dazz-retro-camera（MIT）。拟合时曲线参数做跨通道共享约束，色彩性格主要由串扰矩阵/分离色调/饱和度承载，方法见 `tests/calibrate_stocks.py`。

| 预设 | 气质 | 适合题材 | 标定参考 |
|---|---|---|---|
| 清新人像 (Portra 400) | 低反差、暖调、肤色柔和 | 人像、婚礼、日常 | kodak_portra_400.cube（G'MIC/Pat David 集） |
| 日系空气感 (Fuji 400H) | 轻盈通透、暗部微青、黑位抬起 | 小清新、街拍 | fuji_400h.cube（G'MIC/Pat David 集） |
| 暖阳金调 (Kodak Gold) | 金黄暖调、中等反差 | 阳光、旅行、复古 | cpm35_kodak_gold_warm.cube（dazz-retro-camera） |
| 冷调电影感 (Vision3 500T) | 青橙对比、冷调暗部 | 夜景、都市、电影感 | kodak_vision3_500t.cube |
| 浓郁鲜艳 (Ektar) | 高饱和、高反差 | 风光、静物 | Neg_Kodak_Ektar_100.cube |
| 黑白纪实 (Tri-X) | 高反差黑白、灰轴中性 | 纪实、街拍、特写 | BW_Kodak_TriX_400.cube |
| 自定义 | 中性基础曲线 | 只用手动参数自己调 | （未参与标定） |

### FilmLook LUT 应用

支持两种格式：标准 `.cube`（LUT_1D_SIZE / LUT_3D_SIZE）和 **HaldCLUT `.png`**（8/16-bit RGB，level 自动识别）。`luts/` 目录自带上述 6 个参考 `.cube`（MIT 协议，作者 PopTech Studio），可直接在下拉框选用。

| 参数 | 默认 | 说明 |
|---|---|---|
| lut | 自定义路径 | luts 文件夹中的所有 `.cube` / `.png` 文件 + 自定义路径 |
| strength | 1.0 | 效果强度，0 为原图 |
| custom_path | （空） | 选择"自定义路径"时填写的 `.cube` / `.png` 完整路径 |

### FilmLook 光晕 (Halation)

| 参数 | 默认 | 说明 |
|---|---|---|
| strength | 0.5 | 光晕强度 |
| radius | 标准 | 光晕范围：轻微 / 标准 / 明显 / 夸张 |
| auto_threshold | 开 | 自动检测画面高光区域 |
| threshold | 0.85 | 手动高光阈值，仅关闭自动检测时生效，越低光晕区域越多 |

### FilmLook 胶片颗粒

| 参数 | 默认 | 说明 |
|---|---|---|
| grain_level | 标准 (ISO 400) | 颗粒粗细：极细 (ISO 100) → 极粗 (ISO 3200) |
| grain_type | 亮度颗粒 (保色彩) | 亮度颗粒保色彩；彩色颗粒更有胶片味但会轻微偏色 |
| strength | 1.0 | 颗粒强度 |
| quality | 均衡 | 渲染质量：速度优先 / 均衡 / 画质优先 |
| sharpen | 0.3 | 锐度补偿：颗粒的光学特性会轻微柔化画面，此项加回线条锐度；0 关闭 |
| seed | 0 | 随机种子；同一批次每帧自动重新采样，视频不会出现"静止颗粒" |

### FilmLook 一键胶片

| 参数 | 默认 | 说明 |
|---|---|---|
| preset | 清新人像 (Portra 400) | 同"胶片调色" |
| grain_level | 标准 (ISO 400) | 同"胶片颗粒" |
| strength | 0.85 | 整体强度，同时作用于调色与颗粒 |
| halation_strength | 0.4 | 光晕强度（标准范围、自动检测高光） |
| auto_exposure | 开 | 智能曝光（含高光保护）；画面过曝时可关闭 |
| preserve_luminance | 0.5 | 亮度保持：0 自由改变亮度，1 亮度完全贴回原图；遇到过曝时调高 |
| style_boost | 1.0 | 风格浓度：1 为忠实胶片效果，>1 把风格放大到超出真实胶片 |
| diffusion | 0.2 | 柔光强度：高光弥散的柔焦感（黑柔镜效果），0 关闭 |
| vignette | 0.2 | 暗角强度：四角向中心压暗，0 关闭 |
| seed | 0 | 随机种子 |

## 常见问题

**Q：运行"胶片颗粒 / 一键胶片"时报错，提示缺少 silvergrain 库？**
A：在 ComfyUI 使用的 Python 环境里运行 `pip install silvergrain[gpu]` 后重启 ComfyUI。便携版通常是：
```
D:\ComfyUI_windows_portable\python_embeded\python.exe -m pip install silvergrain[gpu]
```
没有 NVIDIA 显卡可以装 CPU 版（`pip install silvergrain`），但非常慢，请做好心理准备。

**Q：LUT 下拉框里没有我的文件？**
A：确认文件扩展名是 `.cube` 或 `.png`（HaldCLUT）、放在插件的 `luts/` 文件夹里，然后重启 ComfyUI（或刷新节点定义）。16-bit HaldCLUT 只支持非隔行扫描的 RGB/RGBA PNG。

**Q：颗粒在视频批次里每帧都一样吗？**
A：不会。同一批次中每一帧都会独立重新采样颗粒（若 silvergrain 支持 seed 参数，则按 `seed + 帧号` 传入），符合真实胶片颗粒"时间独立"的特性。

## 许可

本插件代码可自由使用与修改；颗粒功能依赖的 silvergrain 为 **AGPLv3** 协议，集成或商用时请注意其传染性条款。`luts/` 目录中 6 个参考 LUT（kodak_vision3_500t / kodak_trix_400_bw / kodak_ektachrome_100d / fuji_eterna_250d / 16mm_reversal_vintage / faded_film_aged）来自 poptechstudio/ai-cinema-studio-engine，**MIT 协议**，作者 PopTech Studio。胶片预设参数由上述参考 LUT 拟合标定，与胶片厂商无关。
