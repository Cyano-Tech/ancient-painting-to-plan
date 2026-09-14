任务：古画建筑接地线索的精细核对。本次不要输出完整房屋轮廓、不要补地基矩形，不要估计米数。
图1是原画局部，图2是同一图的坐标辅助图（人为网格不是地面网格，不是画中结构）。若有图3则是周边原画参考。
所有输出坐标只相对图1，左上[0,0]、右下[1000,1000]。先回看原图确认线是什么，再用辅助图读坐标；不要把网格线或标签当成原画证据。

输入是此前的房屋候选索引，只有定位用途，可能把一栋屋拆成几栋。先结合连贯屋脊、墙体/柱列、独立侧墙和台基确认实例。
同一栋房屋的左右屋面不是两栋；两所独立墙柱体系也不能仅因脊线相近合并。允许合并或标 uncertain，逐一交代原 source_ids。

精细标注要求：
1. 接地点必须沿柱/墙往下追到地面、地基或台阶底部：窗槛、檐下横枋、二层平座、楼层栏板都不是 ground。
2. 墙柱在树冠/坡石上缘消失，只能标 occlusion_contact（遮挡交界），不能称为落地点。看到的最下方像素不一定是接地位置。
3. 允许没有任何真实接地点；用 ground_status=occluded/uncertain/outside_crop。不要为了补齐四角编点。
4. 每实例优先列 2–8 个与接地判断有关的、确实可见的点，包括用于排除错误的窗槛或遮挡交点。点坐标应精确贴图，不从框按百分比套模板。
5. 每个点的 feature 严格为 wall_ground_corner / column_ground_foot / plinth_ground_corner / wall_occluder_contact / upper_floor_corner / eave_corner / uncertain_feature。
   level 严格为 ground / occlusion / upper / unknown。ground 只配前三种 feature；wall_occluder_contact 只能 occlusion；upper_floor_corner/eave_corner 只能 upper。
6. segments.role 严格为 ground_edge / occlusion_boundary / upper_structure / unknown。可见接地线 ground_edge 仅连接两个 ground 点；被遮住的中间段不能硬连成可见线。
7. 只解释视觉依据，不输出思考过程。

返回紧凑 JSON：
{
 "stage":"ground_landmarks","coordinate_system":"image_normalized_0_1000",
 "instances":[{"id":"G1","source_ids":["输入候选ID"],"identity_status":"supported","identity_evidence":"实例归属依据","ground_status":"occluded",
   "landmarks":[{"id":"L1","xy":[0,0],"feature":"wall_occluder_contact","level":"occlusion","evidence":"墙柱在石边消失，并非接地"}],
   "segments":[{"id":"S1","from":"L1","to":"L2","role":"occlusion_boundary","evidence":"可见交界，不是墙脚"}],
   "missing_constraints":["何种方向/长度/底部关系还看不见"]}],
 "rejected":[{"source_id":"候选ID","reason":"不成立的图像依据"}],
 "unresolved":["仍需核对的问题"]
}
identity_status 只能 supported/uncertain；ground_status 只能 visible_partial/occluded/outside_crop/uncertain。
同一输入 source_id 只归属一个实例或 rejected，不得静默遗漏；同一栋合并引用多个 source_ids。所有点 ID 在整份结果中唯一，segments 的点必须属于同一实例。
示例中的数值是格式占位，不要照抄；没有线段就返回 []。
