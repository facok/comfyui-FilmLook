# 胶片数据手册数据（第三方）

`tools/filmdata/` 下的 `.py` 文件是**胶片数据手册的数字化数据**，取自
[JanLohse/spectral_film_lut](https://github.com/JanLohse/spectral_film_lut)，**MIT 协议**，
作者 Jan Lohse。原始数据来自柯达/富士公开的数据手册（sensitometric curves、
spectral sensitivity、spectral density 等）。

| 文件 | 用途 |
|---|---|
| `kodak_portra_400.py` | 负片 Kodak Portra 400 |
| `fuji_pro_400h.py` | 负片 Fuji Pro 400H |
| `kodak_gold_200.py` | 负片 Kodak Gold 200 |
| `kodak_5219.py` | 负片 Kodak Vision3 500T（5219） |
| `kodak_ektar_100.py` | 负片 Kodak Ektar 100 |
| `kodak_trix_400.py` | 黑白负片 Kodak Tri-X 400 |
| `kodak_2383.py` | 印片 Kodak Vision 2383（彩色放映拷贝） |
| `kodak_2302.py` | 黑白印片 Kodak 2302 |

这些文件**不参与插件运行时**，只被 `tools/gen_film_luts.py` 读取以离线生成 LUT。
文件仅包含数据（`FilmData(...)` 构造调用），不含原项目的算法代码；算法部分由本项目
按 `gen_film_luts.py` 文档字符串所述的链路自行实现。

许可：MIT（见原仓库）。分发本插件的 `.cube` 产物时，建议一并保留本说明与原作者署名。
