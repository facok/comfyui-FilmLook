# 预设 LUT 来源与许可

`presets/` 下的 6 条 `.cube` 是随插件分发的参考 LUT，用于 FilmLookColor 节点的胶片预设。
**它们不是本插件原创数据**，分发/再分发时请遵守各自许可。

| 文件 | 对应预设 | 来源 | 许可 |
|---|---|---|---|
| `portra_400.cube` | 清新人像 (Portra 400) | [YahiaAngelo/Film-Luts](https://github.com/YahiaAngelo/Film-Luts) 分发的 G'MIC / Natron Pat David 胶片模拟集 | **CC BY-SA** |
| `fuji_400h.cube` | 日系空气感 (Fuji 400H) | 同上 | **CC BY-SA** |
| `ektar_100.cube` | 浓郁鲜艳 (Ektar) | G'MIC 生成（文件头 `Created by: G'MIC`） | **CC BY-SA** |
| `trix_400.cube` | 黑白纪实 (Tri-X) | G'MIC 生成（文件头 `Created by: G'MIC`） | **CC BY-SA** |
| `kodak_gold_200_warm.cube` | 暖阳金调 (Kodak Gold) | [dazz-retro-camera](https://github.com/ganjmeng/dazz-retro-camera)（`flutter_app/assets/lut/cameras/`，仓库根 LICENSE 为 MIT） | MIT |
| `vision3_500t.cube` | 冷调电影感 (Vision3 500T) | [poptechstudio/ai-cinema-studio-engine](https://github.com/poptechstudio/ai-cinema-studio-engine) | MIT |

## 注意：CC BY-SA 的传染性

上表前 4 条来自 G'MIC / Pat David 的胶片模拟数据集，采用 **CC BY-SA**。如果你要把本插件
（含这些数据）**公开分发**，请自行确认：

1. 是否需要在显著位置署名原作者（Pat David / G'MIC film emulation）；
2. BY-SA 的「相同方式共享」是否会被解释为要求本插件的数据部分同样以 BY-SA 发布。

插件代码本身不受影响；受约束的是这 4 个数据文件。如果不想承担这个约束，可以：

- 只保留 `kodak_gold_200_warm.cube` 与 `vision3_500t.cube`（两条 MIT），其余预设暂时移除；
- 或者用 `tools/gen_film_luts.py`（B 阶段，实验性）从胶片数据手册重新生成同名预设。

## 数据格式

均为标准 `.cube`（`LUT_3D_SIZE 13 / 17 / 33`，`DOMAIN 0..1`），由 `film_core.parse_cube`
解析，按输入 sRGB 显示值三线性插值查表。

## 与 `luts/` 目录的区别

`luts/` 是**用户自行收集**的 LUT 库（默认带 125 个 `.cube`，可自由增删），供
FilmLookLUT 节点的下拉框使用；它的来源与许可由收集者负责，与本文件列出的预设无关。
