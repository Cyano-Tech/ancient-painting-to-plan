# B · Village and Temples in Jiangnan

完整图像来源、馆藏信息及 SHA-256 见 [素材说明](../../../docs/ATTRIBUTION.md)。
`plan.json` 为既有审阅结果的脱敏快照，数值几何未因打包改变；`source.path` 相对该文件所在目录。
`historical_review.json` 是原始计划的历史模型审阅，不是对脱敏文件进行的一次新审阅。

当前回归数据为 23 个占地区域（20 house + 3 other）、3 个未拆清组团、1 个无地基的候选 H29；
35 个初始分块候选恰好归属或拒绝一次。身份复核排除了 H13 的栏墙误报与 H14 的倒影。
这些计数不是已经核实的真实栋数；水岸、隐蔽结构和树根位置仍是近似。

从仓库根目录运行：

```bash
python scripts/replay_examples.py foundation2d --output artifacts/example-b
```

回放后在 `artifacts/example-b/` 查看交互页面与对照图。旧 H14 修订图可从仓库 v0.1.0 历史版本找回。
这些数据为离线测试夹具；当前展示目录只保留十张新的 2D PNG。
