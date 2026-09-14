任务：依据原画与经过一轮复核的可见对象清单，建立“玻璃视图”的结构线与地面接触候选。
这不是纹理生成或真实隐藏结构恢复。只输出有图像依据的可见线，和明确标为推测的隐藏线。不要重新发明场景或强迫统一房屋比例。

先核对清单；明显错误或证据不足的对象 status=deferred 并说明原因，不要硬补。保留输入 ID 和类别，不重编号。建筑只露角时，可参考同院落相近尺度、相似类型的房屋，但要记录 reference_ids 与假设；不够就留未知。

本任务确实需要提出有依据的遮挡补全假设，不是只摘取几条可见线：如果本屋的开间、墙向和相似邻屋提供约束，应尝试给出低置信隐藏边并说明参照；若纵深确实无法约束，再留未知。无隐藏部分时不必人为添加。
精度要求：沿真实脊、檐、墙脚逐折点定位，不要从 bbox 四角内缩若干像素后套用“浅人字+L形”的统一模板。可见屋脊可能倾斜或转折；屋檐弯曲可用多点折线，不能把不同朝向画成水平。

逐对象要求：
- visible_paths 是原图直接可见的结构折线。建筑可画檐、脊、墙边、柱脚；树画树干根部与必要轮廓；地形画真实可见边界。不把外接框四边当作物体轮廓。
- hidden_paths 只画被遮挡结构的假设线，与可见线分开。无需假装补齐每个对象；没有合理依据就 []。
- 建筑地基依据墙脚/柱脚/台基，不是屋顶框。树用根颈落地点；树丛无法定位单根就留 unknown。山脚不能用山峰轮廓替代。水/平地依据岸界/地面界。
- footprint 记录地面接触几何：kind 为 polygon/point/polyline/unknown；points 为坐标。polygon 不重复末尾首点，edge_status 与顶点数相同，分别对应 points[i]→points[(i+1)%n]；polyline 的 edge_status 比点数少一；point 有一个点和一个表示点来源的 edge_status。边/点状态只能 visible/inferred/unknown。未知项用空点、空状态。
- 只有一段墙脚可见时优先 polyline，合理补足时才 polygon。不可把看不见的闭合边标成 visible。多层楼的上层檐口/栏杆不是地面。
- 有支承面证据才给 support_plane，不能只按图像上下分层。先不估实际米数、统一俯角或全图网格。
- 列出 occluded_by，仅原画可见遮挡证据可支持；某像素有树不自动说明树在房前。

输出紧凑 JSON，每个对象一段，尽量控制线条数量（每条 2–10 点），每段说明≤50汉字。必须逐项交代所有输入对象：
{
 "stage":"amodal_structure",
 "coordinate_system":"image_normalized_0_1000",
 "objects":[{
   "id":"沿用输入ID","category":"house","status":"proposed",
   "visible_paths":[[[100,100],[200,110]]],
   "hidden_paths":[[[100,120],[180,130]]],
   "footprint":{"kind":"polygon","points":[[100,120],[180,130],[170,170],[90,160]],"edge_status":["inferred","visible","visible","inferred"],"confidence":"low","evidence":"接地与推测依据"},
   "support_plane":null,"occluded_by":[],"reference_ids":[],
   "assumption":"隐藏结构采用何种推测或为何不补","confidence":"low"
 }],
 "support_planes":[{"id":"P1","evidence":"可见连续台基等依据","confidence":"low"}],
 "view_observations":[{"kind":"ground_direction","observed_line":[[100,100],[200,110]],"evidence":"来自哪一对象的可见墙脚"}],
 "unresolved":["不能可靠恢复之处"]
}
status 只能 proposed/deferred；deferred 对象仍可保留真实可见线，但 hidden_paths 必须为空，footprint 必须 unknown。
每一点是本次输入图像的 0–1000 归一化坐标；边界外不猜测坐标。禁止输出思考过程。
