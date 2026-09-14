# 2D 展示图集 · 10 张 4K PNG

[`examples/`](../examples/) 只放十张 **3840 × 2160、16:9 PNG**，没有幻灯片、3D 预览、JSON 或中间输出。
左侧是输入画作与地基编号，右侧是相对俯视布局。十张来自不同作品，不是同一张图换皮或切成十份。
横卷输入中有馆方细节图/已有选段，不宣称全部覆盖原作长卷。中文大标题为展示主题，不替代馆藏正式名称。

## 图集与原作出处

下列馆藏页面均标记 Public Domain，核查日期为 2026-09-14。作者的“传”“仿”“旧传”限定保留，
不将下载文件名当作权威署名。标注和俯视推断是本项目的研究输出，并非馆方标注或认可。

| PNG | 馆藏作品与来源 | 作者/年代（按馆方） |
|---|---|---|
| [01 桂荫书斋](../examples/01_cassia_studio.png) | [The Cassia Grove Studio](https://www.metmuseum.org/art/collection/search/44601) | 文徵明，约 1532 年 |
| [02 柳岸亭居](../examples/02_willow_pavilion.png) | [Landscape with Pavilion and Willows](https://www.metmuseum.org/art/collection/search/53601) | 传沈周，明或清 |
| [03 帆影江村](../examples/03_river_village.png) | [Landscape with Sailboats](https://www.metmuseum.org/art/collection/search/53603) | 传沈周，明或清 |
| [04 溪桥山居](../examples/04_stream_bridge.png) | [Landscape with Man Crossing Bridge](https://www.metmuseum.org/art/collection/search/53602) | 传沈周，明或清 |
| [05 湖畔亭榭](../examples/05_lakeshore_pavilion.png) | [Pavilion by Lake Shore](https://www.metmuseum.org/art/collection/search/48884) | 佚名，仿文嘉；17 世纪或更晚 |
| [06 青绿园林](../examples/06_garden_estate.png) | [Garden estate](https://www.metmuseum.org/art/collection/search/51575) | 佚名，刘松年伪款；17 世纪 |
| [07 兰亭山水](../examples/07_orchid_pavilion.png) | [Gathering at the Orchid Pavilion](https://www.metmuseum.org/art/collection/search/51394) | 佚名，旧传李公麟；16 世纪 |
| [08 岳阳楼阁](../examples/08_yueyang_pavilion.png) | [The Immortal Lü Dongbin Appearing over the Yueyang Pavilion](https://www.metmuseum.org/art/collection/search/45677) | 佚名，15—16 世纪 |
| [09 秋山梵宇](../examples/09_autumn_temples.png) | [Buddhist Temples amid Autumn Mountains](https://www.metmuseum.org/art/collection/search/40106) | 佚名，仿燕文贵；14—15 世纪 |
| [10 雪江村舍](../examples/10_snowy_retreat.png) | [Snowy landscape with rustic riverside retreat](https://www.metmuseum.org/art/collection/search/40987) | 传刘松年，旧传高克明；约 12 世纪末 |

## 如何解读

- 褐色为房屋支承地基，灰绿色建筑为 `other`；永久亭阁可以属于 `house`，桥、廊和围挡不冒充住宅。
- 箭头指建筑正面，不是屋脊长轴。虚线四角包含遮挡补全假设；不是由图像包围框直接换算的长宽。
- 黄色是局部支承平地，可以叠置于山体上。绿色是山地占地示意，蓝色是有证据的水域，树只画代表性根点。
- 未分类底色不等于已确认的平地。山地内的纹线、背景网格只用于图面阅读，不是测得的等高线、米制格网或相机标定。
- `*` 表示未拆清组团，不等于一栋；问号候选不拥有地基。占地区域数量不等于真实栋数。

## 生成与边界

本批使用云端 `qwen3.8-max` 做全图地形分析、重叠局部巡查、布局与视觉审阅，本地只执行 CPU 图像处理、
schema 校验、有限中心位置优化和 SVG→PNG 排版，没有使用本地 GPU。
`scripts/render_example_cards.py` 不修改模型的地基坐标、长宽或朝向，只调整显示区域、样式和文字。

这是为展示而选择的十个案例，不是随机抽样评测集；模型自检、几何检查及人工目视审阅均不等于独立真值验证。
水岸、隐藏墙脚、树根和远近尺度仍是近似推断，不宜当作精确测绘图。

展示只分发 PNG。原始请求、私有日志与本批工作数据保留在本地，不上传。
仓库的离线回放命令仍使用 `tests/fixtures/` 中的历史快照，**不声称能仅凭这十张 PNG 复现新推理**。
两套方法的源码、提示词、测试和旧夹具均保留；旧预览可从 v0.1.0 历史版本恢复。
