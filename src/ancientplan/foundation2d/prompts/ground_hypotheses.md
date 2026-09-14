任务：从原画和逐点接地线索生成建筑地基的“有证据约束的假设”。最终目标是二维俯视图，但不是实测复原。
图1为原画局部，辅助坐标图不是原画结构。如有周边参考图，仅用于比较近邻建筑风格与比例；不能直接拷贝不同远近建筑的像素尺寸。
所有 image_corners 仍相对图1的 0–1000 坐标。不要将屋顶框、窗槛、栏杆、树冠边界直接作为地基。

如果输入包含 previous_hypotheses 和 previous_check，这是修订任务：重新对照原图检查，不照抄先前“通过”的文字。geometric_correspondence_checks 只是原坐标边长的机械计算，不是实际地面长度。
特别注意：若先前角点顺序把短侧墙当成 A→B，却按长前墙给了宽/深比，必须在 image_corners 中真正重排角点，不能只在解释中把A口头换成其他角。比较前墙主要门窗与侧墙方向，给出完整修订结果，不返回修改补丁。

重要：用户允许根据本屋结构和相似邻屋推测被遮挡部分。不要因有遮挡就一律空白；有形状/方向依据时给出最多两种低置信候选，并解释假设。
但必须区分“俯视形状可估”与“能放到画面哪里”：没有定位依据时可返回 shape_only，只给长宽比范围，不编造确定的接地坐标。
墙下缘在石边消失只是遮挡交点，不能当真实墙脚。多层阁楼必须追到最底层接地，不能把二楼平座当地基。

每实例只能使用输入 instances 的 ID；保持合并后的身份，不重新拆成多个房子。
为每个输入实例输出一个记录：
{
 "stage":"ground_hypotheses","coordinate_system":"image_normalized_0_1000",
 "instances":[{"id":"G1","status":"proposed","alternatives":[{
   "id":"P1","placement":"inferred",
   "image_corners":[[0,0],[1,0],[1,1],[0,1]],
   "vertex_status":["inferred","inferred","inferred","inferred"],
   "vertex_landmark_ids":[null,null,null,null],
   "edge_status":["inferred","inferred","inferred","inferred"],
   "width_depth_ratio_range":[1,2],
   "reference_instance_ids":[],"evidence_landmark_ids":[],
   "placement_basis":"墙高、原图可见段、遮挡和参照如何限制位置",
   "ratio_basis":"为什么判断该相对长宽比范围，哪些来自场景哪些是假设",
   "uncertainty":"未知的范围与可能错位原因","confidence":"low"
 }],"reason":"为何可以提候选或仍缺约束"}],
 "cross_instance_warnings":["重复归属、候选互穿、不同支承面等问题"]
}

status 只能 proposed/insufficient。insufficient 对象 alternatives=[]。
方案 ID 只需在所属实例内唯一；全局由 instance_id + alternative_id 联合标识。
placement 只能 anchored/inferred/shape_only。
image_corners 按 前墙左端A→前墙右端B→后墙右端C→后墙左端D 的顺序，构成不自交的四边形。前墙依据可见门窗或主要入口，不代表全局正北。
shape_only 时 image_corners、vertex_status、vertex_landmark_ids、edge_status 全部 []，明确位置无法可靠估计。
其他 placement 须四点、四个 vertex_status、四个 vertex_landmark_ids、四个 edge_status。
vertex_status 和 edge_status 只能 visible/inferred；只有原画真正 ground 点才可 visible，必须引用其 landmark ID 且坐标原样使用。
所有可见边中段也必须真实可见，不能跨越遮挡。不能把 upper 或 occlusion 类型点提升为可见地基顶点。
anchored 至少要一个真实 ground 点，其余边可推测；没有这样的点只能 inferred 或 shape_only。
evidence_landmark_ids 可以包括用于尺寸/方向参考的 upper 或 occlusion 点，但不把它们当作接地点。
width_depth_ratio_range 是地基前墙宽/进深的正数区间，不是图片框长宽比；不同房屋依据不同证据，不强迫固定比例。不能约束比例则 insufficient。
只输出 JSON，不要思考过程；数值示例只是格式占位，必须从本次图像判断。
