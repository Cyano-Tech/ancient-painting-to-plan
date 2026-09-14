# 方法 A：检测缓存 → 简化 3D → 正交俯视

## 实现范围

这条路线最初用于三个局部古画场景。当前发布版将其中的通用算法和示例先验分开：
代码不再根据示例文件名选择手写房屋，也不会在检测缺失时回退到人工房屋坐标。
示例的相机、山地、水域、树丛、遮挡关系仍是人为给定的研究先验，位于
[`scene_priors.json`](../tests/fixtures/cuboid3d/scene_priors.json)。把先验放进数据文件不等于自动识别了它们。

## 算法步骤

1. **整栋定位（历史推理阶段）**：Qwen3-VL-8B 全图和重叠分块提出建筑候选；YOLO 提供子结构证据，SAM 2.1 给出可见区域，RGB 分类器和包含关系规则剔除明显非建筑、分解组团。独立建筑不应等同于每一片屋顶或遮阳篷。
2. **独立方向判断**：Qwen3-VL-32B 看同一个已接纳建筑的隔离局部，找长屋脊/长檐、可见短端墙及其底角。只推断既有 ID 的方向，不增删实例。沿长轴指向可见短端墙的方向转成 yaw。
3. **尺寸估计**：长边先来自可见区域的保守包络；短边优先使用可信的短端墙地面角点，经相机方向投影求深度。证据不足时，在固定方向下对 SAM 轮廓作离散长宽比拟合；无法分辨时使用 3:1 先验。可见框宽高不是最终俯视长宽。
4. **体块与放置**：按显式正交相机求长方体；使用统一缩放/位置调整满足非重叠、避水等约束。尺寸推断后锁定比例，不通过单独压扁短边掩盖冲突。房屋均放在 z=0；山、水、树为粗略代理几何。
5. **导出**：原视角语义/普通渲染、90° 正交俯视、透视调试图、glTF 2.0 GLB、结构化 scene/metrics。

## 容易误读的指标

`raw_semantic.png` 保留了历史文件名，但实际是依据建筑体块假设生成的初步面图，带合成边界扰动，**不是从原画独立分割出来的真实面标注**。
`geometry_consistent_semantic.png` 与 `original_view_semantic_render.png` 来自同一几何和渲染器。
因此 `house_semantic_silhouette_iou` 是内部一致性检查，即使为 1.0，也不能说明原画重建准确。
发布版不再输出旧代码中硬写为 1/0 的逐面准确率/距离值。

场景相机不是经过独立标定的真实相机；没有实现原设计设想中的通用“共同优化任意古画相机和所有物体”。
不支持真实山顶高度、复杂屋顶、逐木构件、米制测量或经过验证的透视恢复。

## 离线执行

在仓库根目录、安装基础依赖后：

```bash
python -m ancientplan.cuboid3d.pipeline \
  --input-dir tests/fixtures/cuboid3d/inputs \
  --scene-config tests/fixtures/cuboid3d/scene_priors.json \
  --detections-dir tests/fixtures/cuboid3d/house_detection \
  --orientations-dir tests/fixtures/cuboid3d/building_orientation \
  --output-dir artifacts/cuboids
python -m ancientplan.cuboid3d.validate_outputs artifacts/cuboids
```

`--input` 可选，指定其中一张图片。不存在缓存、图像尺寸/哈希不符、方向缓存未绑定当前检测 JSON 时会拒绝执行。
示例目录名用于查找对应数据文件，不承载按图修正的代码逻辑；测试会将同一个缓存改名验证这一点。

## 如果要对新图重新检测

以下三个源码入口保留供研究，不属于默认安装/离线测试路径：

```text
ancientplan.cuboid3d.detect_whole_buildings_vlm
ancientplan.cuboid3d.detect_building_instances
ancientplan.cuboid3d.infer_building_orientations_vlm
```

它们需要另行安装兼容的 PyTorch、Transformers、Ultralytics、OpenCV、NumPy、Accelerate 等，并准备检测器与分类器权重。
本仓库**没有这些自训练权重**，也没有声称能仅靠 `pip install` 复原历史训练环境。
`--device` 必须明确指定；默认回放和 CI 都不会运行这些入口。FP8 32B 模型的硬件/量化兼容性需要在单独获准的机器上验证。

输入链条示意（不是可直接复制的完整命令，所有大写项需提供实际文件）：

```text
detect_whole_buildings_vlm IMAGE --device DEVICE --output-dir GROUNDING
detect_building_instances IMAGE --detector DETECTOR.pt --classifier CLASSIFIER.pt
    --device DEVICE ...
infer_building_orientations_vlm IMAGE DETECTION.json --device DEVICE ...
pipeline --input IMAGE --scene-config PRIORS.json --detections-dir ... --orientations-dir ...
```

检测脚本的具体 proposal 参数见 `main()`；不提供未经当前环境验证的训练命令或伪造权重下载地址。
`prepare_building_dataset.py` 与 `prepare_building_classifier_dataset.py` 保存了历史数据转换方法，原训练数据不随仓库分发。
相机/地形先验仍需另备；仅运行检测器不足以把这条旧路线变成通用全图重建器。
