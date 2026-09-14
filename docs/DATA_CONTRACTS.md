# 数据结构与坐标约定

## A：cuboid3d

`scene_priors.json` 的 `scenes` 以输入文件 stem 索引相机/地形先验；无 `houses` 手工坐标。
缓存检测 JSON 为 `automatic_whole_building_instance_parsing`，包含实例 ID、像素 mask 框和轮廓。
方向 JSON 为 `automatic_building_orientation_inference`，携带原图 SHA-256 与检测 JSON SHA-256。
图像尺寸相同并不能证明是同一张图，两个哈希都必须匹配。

- 原画像素：左上为原点，x 向右，y 向下。
- 世界：x 向画面右、y 向远处、z 向上，单位任意相对值；房屋底 z=0。
- 方向：`top_view_angle_deg_clockwise_from_right` 是沿长轴指向可见短端墙的角度，转换为 world yaw；不是 B 的正门朝向。
- `length/depth/height` 为体块尺寸；长度必须大于深度是旧方法的建模约束，不是所有真实建筑的普遍事实。
- 输出契约及 JSON Schema 见 [`contracts`](../src/ancientplan/cuboid3d/contracts)。

## B：foundation2d

源图坐标与平面坐标必须分开，不能把原画可见框当成俯视地基。

| 字段 | 含义 |
|---|---|
| `source.path/size/sha256` | 源图定位、EXIF 方向修正后尺寸、原始字节哈希；发布示例 path 相对 plan.json |
| `source_bbox` | 原画归一化可见范围 `[x0,y0,x1,y1]`，每轴 0–1000 |
| `source_footprint` | 原画上的推测地基四边形；不等于可见轮廓或像素真值 |
| `source_anchor` | 地面代表点；独立复核后可由地基中心推得，不从屋顶取点 |
| `plan_center` | 俯视平面中心；不是米制坐标 |
| `plan_size` | `[width,depth]`，width 沿正面，depth 沿正面法向 |
| `front_clock` | 正面法向的钟点方向；12 向上、3 向右、6 向下、9 向左，可有小数 |
| `ratio_range` | 允许的 width/depth 比例区间；不从 bbox 宽高直接抄来 |
| `source_ids` | 归属的初始候选；每个初始 ID 恰好归属或被拒绝一次 |
| `observation_lock` | 独立复核锁定的原画字段；布局步骤不可覆盖 |
| `grouped/unit_count_range` | 未拆清组团及数量区间，不冒充一栋确定建筑 |
| `pending_candidates` | 只显示位置的候选，不能携带已确认地基/面积 |
| `evidence/assumption/confidence` | 可见支持、补全假设、模型置信表述；不是校准概率 |

`terrain` 包含山、水等区域、局部地面 zones、树根代表点、路径与总体范围。
台地 `flat` 可以处于山顶；这里标的是承载关系，不代表绝对海拔零。
`ground_axes_image` 是局部原画地面方向假设，不是相机标定结果。

JSON 校验器见 [`plan_schema.py`](../src/ancientplan/foundation2d/plan_schema.py) 和
[`scene_schema.py`](../src/ancientplan/foundation2d/scene_schema.py)；请求阶段说明在 [`prompts`](../src/ancientplan/foundation2d/prompts)。

## 脱敏与历史证据

导出的缓存/计划保留数值几何与模型文本证据，但本机路径被替换为相对路径或 `archive/...` 标签。
`archive/...` 是历史出处标识，不保证该运行目录随仓库分发；本包不是全量 API 追踪档案。
`release_export` 记录导出前哈希和修改范围。
A 的方向缓存重新绑定脱敏后的检测 JSON 哈希，并单独保留旧绑定；不是一次新的模型推理。
B 的 `historical_review.json` 审阅的是原始计划哈希，不是脱敏计划的新字节，因此不能把它当作新发布快照的在线认证。
几何数值未因打包而调整；回放报告明确记录 `historical_review_rerun=false`。
