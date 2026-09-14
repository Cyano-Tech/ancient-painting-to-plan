# A · 三个局部场景的离线 3D 回放

`inputs/` 是原始交接裁片；图像公开分发条件尚未完整确认，见 [素材说明](../../docs/ATTRIBUTION.md)。
`house_detection/` 和 `building_orientation/` 是脱敏后的历史模型结果，`scene_priors.json` 是显式人工相机/地形先验。
旧的手写房屋列表不随先验导出，也不作为缺失缓存时的替代检测。

固定回归预期：`scene_01_compound` 4 个、`scene_02_pond_house` 2 个、`scene_03_hillside_hamlet` 2 个缓存房屋实例。
这不是独立数据集上的召回率或精确率。三场景不同于 B 的整幅 Met 示例。

从仓库根目录运行：

```bash
python scripts/replay_examples.py cuboid3d --output artifacts/example-a
```

预生成结果见 [preview](preview)，含三个 GLB、场景 JSON、图片和回放报告。
缓存里的 `archive/...` 表示未随包分发的历史路径；回放仅使用当前目录中的实际输入与缓存文件。
