# 方法 B：云端多模态 → 地基假设 → 直接俯视

## 从“物体图像”到“地面占地”

目标不是把原画矩形框压到平面，而是回答：**每个实体在地面上占据哪里？**
先按 `house / tree / mountain / water / flat / other` 理解地面相关元素。
山顶台地也可以是 `flat`：这里表达的是局部承载面，而不是“海拔必须为零”。
树优先记录树根/树群代表位置，房屋记录地基，山体与水域给出粗略范围；云雾、人物不自动变成房屋或平地。

## 当前可执行的流程

1. **全图加重叠分块**：全图判断地面分区、相对关系；分块寻找建筑候选，防止大画小建筑被缩图遗漏。每个候选 ID 必须被一个实体接收，或明确拒绝，不允许无声丢失/重复归属。
2. **可见证据与遮挡分开**：记录可见框、底边/墙脚、推测地基、证据描述、朝向、比例区间和不确定性。看到一个角不自动变成一栋已知完整房屋。
3. **局部身份复核**：先用没有旧框/旧描述的局部图独立观察，再做 ID 对应。扩大上下文查看屋顶—墙—地面的结构连续性，区分倒影、栏墙、遮阳篷、相邻屋顶和真正独立建筑。
4. **地面与俯视布局**：依据局部地面轴向和地基假设形成共享的相对平面。原画斜网格是局部假设；俯视图是重排后的平面，不是整幅画的唯一逆透视解。
5. **约束优化**：独立观察锁定的原画坐标不能为了“让图好看”被布局步骤改动；独立比例区间不能被整体布局擅自放宽。CPU 优化器只在有限范围移动平面中心，不改宽深、不删除建筑来消除碰撞。
6. **再审阅再发布**：看实际渲染的原画对应图和完整俯视图，检查当前几何事实。未确认候选只有位置问号，没有虚构占地。发布工具检查最终审阅绑定的计划哈希，拒绝未应用的修改或主要几何冲突。

云端阶段、人工指出疑点、独立复核和 CPU 约束共同构成研究工作流。
“持续优化”是有证据的迭代过程，不是一个已证明收敛、可无限付费自动执行的循环。
单次 schema 校验通过或模型说“合理”，都不代表识别真值已被验证。

## 云端配置

先阅读 [SECURITY.md](../SECURITY.md)。每次调用会把所选图像/裁片、提示词和所需上下文发往你配置的服务，可能计费。
本地不会调用 GPU，也不会在云端失败时降级到本地模型或别的模型。

```bash
cp config.example.json config.local.json
# 用编辑器填入你账号对应的 HTTPS base_url 和 api_key_file。
# 密钥文件应保存在仓库外，且权限为 600；不要把密钥填进 JSON 或命令行。
export ANCIENTPLAN_CONFIG="$PWD/config.local.json"
export ANCIENTPLAN_RUNS_DIR="$PWD/artifacts/runs"
python -m ancientplan.foundation2d.qwen_cloud smoke --image tests/fixtures/foundation2d/source.jpg
```

密钥文件格式为一行 `DASHSCOPE_API_KEY='YOUR_KEY'`，可带 `export` 前缀。
程序解析该行，不执行 shell 文件；拒绝非本人所有、组/其他用户可读的文件。
不要把真实 Key 粘贴到 issue、终端历史或提交中。

示例使用历史账号已成功调用的 `qwen3.8-max`，**不是承诺所有账号/地域都支持该标识**。
整幅工作流的接纳检查固定该模型及其同名前缀快照；不要只改配置里的模型名而忽略审阅链。
端点由用户提供，无业务空间专属地址随仓库分发。无隐式重试和重定向。

## 分阶段处理新图

以下第一条命令可直接用自己的图片执行；后续 `SURVEY_MANIFEST`、`PLAN_JSON`、`REVIEW_RUN` 等是前一步实际打印的文件/目录，需替换为真实路径。

```bash
python -m ancientplan.foundation2d.complete_plan survey --image path/to/painting.jpg
python -m ancientplan.foundation2d.complete_plan layout --manifest SURVEY_MANIFEST
python -m ancientplan.foundation2d.render_complete_plan --plan PLAN_JSON --png
python -m ancientplan.foundation2d.complete_plan details --plan PLAN_JSON
python -m ancientplan.foundation2d.complete_plan critique --plan PLAN_JSON --details DETAIL_MANIFEST
python -m ancientplan.foundation2d.complete_plan apply --plan PLAN_JSON --review REVIEW_RUN
```

每步输出独立版本；`apply` 不等同于模型建议全部可信。查看新增的计划文件并重新渲染，再决定是否继续。
对局部身份、尺寸或遗漏有疑点时，使用 `identity_audit`、`apply_identity_evidence`、`apply_shape_evidence`、
`promote_missing` 的显式命令；各模块 `--help` 给出参数，关键 schema 见提示词和验证器。
只在证据成立时提升候选；无法拆清的组团保留 `grouped` 和数量区间。

最后对**当前新版本**重新渲染并复核：

```bash
python -m ancientplan.foundation2d.render_complete_plan --plan FINAL_PLAN_JSON --png
python -m ancientplan.foundation2d.complete_plan final-check --plan FINAL_PLAN_JSON
python -m ancientplan.foundation2d.publish_complete_plan \
  --plan FINAL_PLAN_JSON --review FINAL_REVIEW_RUN --output artifacts/reviewed-publication
```

`publish_complete_plan` 是本地打包，不上传 GitHub；审阅不通过会拒绝发布。
示例回放不需要上述任何云端步骤，且不会冒充新一次模型审阅。

## “玻璃化补全”的含义与尚未做到的事

代码保留早期的 `inventory → review → structure → glass/ground` 探索入口，用 SVG 区分可见/遮挡假设。
完整计划路线将这类判断折叠进结构化地基和局部审查中。
它**没有**通过图像生成去真实补绘被遮挡的原画，也没有声称恢复建筑内部、山体后的真实拓扑。

当前局限包括水岸边界较粗、被遮挡对象可能遗漏、树群不是逐株清点、多局部视点无法唯一标定，
以及长方形地基不能覆盖所有复杂建筑。H14 的倒影修正不等于整幅水面也已精确恢复。
