# 复现与验证范围

## 三种不同的“复现”

1. **离线回放（本发布已验证）**：使用同一输入、缓存观测和计划 JSON，重新运行 CPU 几何/渲染/验证。无需密钥或 GPU。
2. **新图云端推理**：本次 10 张 PNG 来自新调用的 Qwen 3.8 和 CPU 渲染。请求与审阅需要自己的服务授权；模型响应并不保证跨时间确定。`replay_examples.py` 回放的是 `tests/fixtures/` 中的旧快照，不会重新调用云端，也不自动复现这十次新推理。
3. **旧本地训练/推理（研究源码保留，未在本发布重跑）**：A 的权重、完整数据集和 GPU 栈不打包；不能声称仅凭仓库能重现历史训练过程或准确率。

## 可复制的检查命令

从仓库根目录、激活环境后：

```bash
python -m pip install --require-hashes -r requirements.lock
python -m pip install --no-deps -e .
ruff check src tests scripts
ruff format --check src tests scripts
python -m pytest -q
python scripts/replay_examples.py all --output artifacts/verification
python -m ancientplan.cuboid3d.validate_outputs artifacts/verification/cuboid3d
python scripts/check_release.py
python -m build
```

wheel 只包含 Python 包及 prompts/contracts；源代码分发包（sdist）还包含文档、测试、脚本和精选示例。
大图和演示不放入 wheel；从 wheel 单独安装时，需要另外下载仓库/源码包才能回放示例。
展示目录只放十张 PNG；原始请求、私有运行日志和新图的工作文件不上传。旧预览仍可由夹具回放或从 v0.1.0 找回。
依赖锁定文件为 CPU 运行与开发工具集合，不含模型推理框架。系统 Cairo 和字体仍需按平台安装。
首次安装需要网络，pytest 会阻断测试进程的网络连接；GitHub CI 不提供云端凭证。

## 测试覆盖什么

- 云端：端点与密钥文件限制、无重定向、返回模型/完成原因检查、响应脱敏、缺配置提前失败；使用合成响应，不发送真实请求。
- 坐标与 schema：裁片到原图转换、退化四边形、非有限值、比例区间、候选归属、前向钟点约定。
- 证据约束：身份拒绝、反射不能当独立实体、来源几何锁定、组团和未定候选不得假装确定地基。
- 布局：非重叠判定、有限中心移动、不可行时保留问题，不缩房子/删对象来蒙混过关。
- 3D：地面投影往返、方向转换、缓存哈希、缺失缓存不回退、改文件名仍按同一规则处理、GLB/契约/比例锁定。
- 发布：迁移目录后的相对源图路径、SVG/PNG、非空输出保护、固定示例数据回归、隐私和大文件检查。

固定示例测试中的 `4` 或 `H14` 是**回归断言**，不是传给推理的规则。推理源码不根据这些编号/场景坐标选择答案。
初始 v0.1.0 的历史检查记录在 [release-checks.json](release-checks.json)，其中零云调用只描述当时的打包，
不描述后续新图集生成；新提交的实际检查以对应 CI 为准。

## 不能证明什么

没有独立、完整的地基标注集；未报告房屋检测 precision/recall、边界真实误差或米制定位误差。
A 的语义 IoU 是同几何渲染一致性。B 的“零已检测几何冲突”只对当前多边形和检查规则成立，不证明真实场景没有冲突。
示例数、组团数不是检测准确率，模型自检不是独立评测。

图像文件可能因 Pillow/Cairo/字体版本不同出现边缘或文字栅格差异；关键验证以结构化数值、约束和来源绑定为准，不要求 PNG 字节完全一致。
导出时的数值几何没有重新推断。[source_manifest.json](source_manifest.json) 和旧夹具的 `release_export`
记录的是 v0.1.0 原始导入文件的哈希，不是当前源码或这十张新 PNG 的校验清单。

## 发布前仍需的外部决定

本版本按私有研究用途提交。若要公开：先确认代码许可证，核清旧三张裁片的图像出处和使用权，
并复核依赖许可证对目标分发方式的要求。不将尚未提供的权重、缺少独立真值评测等事项藏在“可复现”标签下。
