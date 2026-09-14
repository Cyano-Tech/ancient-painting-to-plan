这是整幅古画二维俯视地基近似图的最后一轮只读验收。第一张原画，图2为实际渲染的共享坐标俯视图，图3为原画对应图。JSON含同一版本的地形、地基、实例来源和已计算的几何检查，不包含旧版计数/越界结论。

判断目标是：该完整近似图是否已足够合理，能交给用户确认整体布局；不是证明真比例/真北/真实相机。逐区检查主要可见建筑和地面元素是否在图中有相应表示，房屋是否明显重复、长宽失常、朝向相反、穿插重叠、中心不在支承面、落到无证据水面。高处平地可叠置在山体占地上，建筑组grouped包络不当独栋，树点为根部代表点非精确株数。允许隐藏地基边界、台地边界、高差及多视角拼合的合理不确定性，不能因这些不可观测事项要求无限循环。

严格检查实际plan_center/plan_size/front_clock形成的矩形，不可拿source_footprint来解释平面碰撞已解决。原画和地基覆盖确有偏差时如实说明，但不要重新把已核实同屋顶的残段单独成栋，也不要用旧错误人物/台基归属推翻共同放大图的结构连续性证据。历史审查记录是参考，不是真值。

输出严格JSON，只报告，不修改任何数据：
{"stage":"plan_final_check","verdict":"ready_for_confirmation|needs_revision",
 "blocking_issues":[{"ids":["Hxx"],"evidence":"仍明显不合理的具体证据","required_change":"需要改什么"}],
 "coverage":[{"region":"远景/左侧/中谷/左前/中前/右前等具体区域","assessment":"对应关系检查结果"}],
 "uncertainties":["不妨碍使用近似布局，但必须让用户知道的推测或未定事项"],
 "summary":"为什么能或不能交给用户确认"}

ready_for_confirmation时blocking_issues必须空；若仍有可明确核实的漏屋、重复屋顶或无解释的重大碰撞，必须needs_revision。几何minor边界越界若由岸边基座/部分遮挡的边界不确定性解释可列uncertainties，不假装精确贴合。不要因为数据形式完整就盲目通过。

若有geometric_facts，计数与角点已经由CPU从当前快照计算。不得自行心算后把旧版数量或已经消除的越界写回来。输出另加verified_counts:{"active_footprint_regions":实际数,"pending_candidates":实际数}，必须逐值匹配输入。grouped只表示一个包络，不可据此宣布准确栋数；pending问号只表示候选位置，不是地基。geometry_checks为空时没有当前代码发现的碰撞/越界，不要编造历史minor；有新的视觉矛盾则指出可见证据。请重点评价视觉对应，不要在summary/coverage重复一长串容易算错的坐标。水中倒影没有第二块实体地基，但排除倒影不代表已经精确恢复其水岸边界。
