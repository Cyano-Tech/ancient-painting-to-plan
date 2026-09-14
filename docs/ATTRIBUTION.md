# 素材、模型与依赖来源

## 示例画作

当前展示目录的十张 2D PNG 对应十件不同馆藏作品，原作来源和完整作者限定词见
[2D 图集来源表](EXAMPLES_2D.md) 和 [结构化出处](example_sources.json)。
下文 A/B 输入属于保留的历史回归夹具，不再是当前 `examples/` 图集。

### B：Village and Temples in Jiangnan

无名氏，明代，15 世纪早期；《樓閣江帆圖》團扇。The Metropolitan Museum of Art，藏品号 **1989.363.43**，
馆方标注 Public Domain；credit line 为 Bequest of John M. Crawford Jr., 1988。
依据：[馆藏原页面](https://www.metmuseum.org/art/collection/search/45665)。

仓库 `tests/fixtures/foundation2d/source.jpg` 是项目已有下载文件，3946 × 3716，SHA-256
`68a345255caf9336f305bc8649b1be8836c1ca9bd7aa759ff9e370258dbb2b59`。
标注、地基、H14 修订示例都是研究生成的解释，不是馆方标注，也不代表馆方认可。

### A：三个历史局部裁片

`scene_01_compound.png`、`scene_02_pond_house.png`、`scene_03_hillside_hamlet.png`
来自项目已有研究交接包。当前可证实的是本项目输入文件和哈希，**尚未完整核实原作馆藏、裁片制作来源及图像再分发条件**。
保留于私有研究包供内部复现；不要据“古画”二字直接宣布所有输入数字图像均为 CC0，或在未确认前公开这一整组示例。
它们和 B 的整幅画不是同一个数据集。

## 模型与软件（未随仓库分发权重）

| 来源 | 本项目用途 | 许可/使用说明 |
|---|---|---|
| [Qwen3-VL-8B](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) | 历史整栋候选 | 以该模型仓库的许可与说明为准 |
| [Qwen3-VL-32B FP8](https://huggingface.co/Qwen/Qwen3-VL-32B-Instruct-FP8) | 历史独立方向与角点 | 模型页标注 Apache-2.0；量化/运行依赖需另配 |
| [SAM 2](https://github.com/facebookresearch/sam2) / SAM 2.1 | 可见区域分割 | 查看上游代码和 checkpoint 许可；本仓库不包含 checkpoint |
| [Ultralytics](https://www.ultralytics.com/license) | 历史 YOLO 检测与分类 | 上游提供 AGPL-3.0 与 Enterprise 路径；本项目许可不替代这些条件 |
| [Chinese Landscape Object Detection](https://www.kaggle.com/datasets/edmundxu/chineselandscapeobjectdetection) | 历史训练数据转换脚本的目标格式 | 不分发数据；须自行核对原页面的当前数据条款及图像来源 |
| [CairoSVG](https://cairosvg.org/documentation/index.html) | CPU SVG→PNG | 可选渲染依赖；系统 Cairo 和字体需单独提供 |
| [Pillow](https://github.com/python-pillow/Pillow), [jsonschema](https://github.com/python-jsonschema/jsonschema) | CPU 图像读写与契约校验 | 依赖保留各自上游许可 |

云端 `qwen3.8-max` 是服务模型标识，不等同于本地 Qwen3-VL 的模型权重。
其账号可用性、服务条款、数据处理和计费以用户选择的服务为准；本项目不转授云端使用额度。
外部页面可能更新；以上来源说明在初始发布整理时核查，不构成法律意见或商业合规保证。
