本次专门放大复核古画里的建筑身份与地基。第一张是原画局部，输入候选只是待核对的旧猜测，可能误框岩石、重复楼层或漏了邻屋。请自己看图，不顺着旧结论说。

所有输出图像坐标均为本张局部图归一化x,y∈[0,1000]。输入bbox/anchor也已转换为当前局部坐标。请逐栋核对可见屋脊、柱墙、平台、与相邻建筑的断开/连续关系；底层地基在墙脚/最低承台，不在屋檐、栏杆或上层墙身。被遮挡的允许推测，但分清依据与猜测。

输出JSON：
{"stage":"plan_details","decisions":[
 {"id":"输入H编号","status":"keep|uncertain|merge|reject","merge_into":null,
 "category":"house|other","subtype":"殿/住宅/塔/亭阁/廊/其他",
 "bbox":[x0,y0,x1,y1],"anchor":[x,y],"footprint_image":[[x,y],[x,y],[x,y],[x,y]],
 "front_clock":6,"ratio_range":[low,high],
 "evidence":"具体看到了什么，尤其说明同栋还是不同栋；假房屋是何种物体",
 "assumption":"地基隐藏边与方向不确定性"}],
 "missing":[{"id":"N01","category":"house|other","subtype":"...","bbox":[...],"anchor":[...],"footprint_image":[...],"front_clock":6,"ratio_range":[low,high],"evidence":"独立可见证据，不得与已保留者重复","assumption":"..."}],
 "coverage_notes":"整张裁片扫描情况"}

每个输入候选id必须恰好有一项decision。status=merge需merge_into指向本批另一个keep/uncertain的ID，不能形成环。若为reject，仍保留输入bbox/anchor/footprint_image，只是不再把它当真建筑。若来源切断房屋，不能以切断为理由断定为两栋。既然观察到整栋且它的分段也是输入，明确合并后不能再重复保留分段。
front_clock表示俯视正面朝向（12上3右6下9左），6只是类型示例，不是默认。请分别判断两侧墙哪一面是主立面，再推朝向；墙长轴/屋脊方向不是朝向。ratio_range为前墙宽/进深，不是图像检测框比例。可见山墙与主立面空间关系也可约束比例。footprint_image需要凸四边形、环绕顺序，四角均是地基假设；不能把山崖整体填进去充当宅基。多栋建筑连续院落若无法拆清，可标uncertain建筑群而不是凭空给确切数量。

若提供 blind_inventory，它是同一原画局部在完全不见旧框/旧描述时的独立识别，不是真值。必须对照屋顶—墙柱—最低支承连续链重新判断；分歧必须解释，不能简单服从任何一轮。矩形分格可能是台基装饰而非窗，腰檐、栏杆、台阶、院墙不能各自数成房屋。仅有水平分隔带不足以认定上下两栋；独立性需要各自屋顶、承重墙与落地关系的证据。若旧框只圈中同栋的下层/台基，应merge，并把保留者bbox覆盖整个可见建筑、地基放到最低支承，不仅删除一个编号了事。
对于输入 identity_audit=true 的请求，每项 decision 还必须输出：
"foundation_trace":"本框涉及的屋顶位置→墙柱→最低支承，与邻居是连续还是独立；哪些实际看不清",
"independence":"independent|shared|nonbuilding|unresolved",
"ground_contact":"visible_partial|inferred",
"bbox_reason":"为什么调整或保留可见范围；不准引用平面图碰撞/zone作为原画坐标依据"。
同时给 label 字段，按新识别的本体写一个简短位置名称，不沿用错误旧名称。

每个候选先判断 physical（实体）、reflection（倒影）或unclear，并输出 appearance_kind 和 reflection_evidence。纸绢色水平带也可能是水面；须同时看上方实物和下方疑似镜像，比较屋脊、檐角、窗格、岸线和遮挡。窗墙在上而屋脊在下是重要倒影线索，不能将倒置的结构强说成独栋廊屋。已确认倒影：status=reject，independence=nonbuilding，merge_into=null（它不是第二栋建筑或同栋楼层），几何只作定位占位，不进入地基图。不确定镜像则uncertain/unclear，不虚假断言为实体。没有倒影时也明确记录检查依据，不可因为提示提到倒影就删除真正房屋。
keep必须有可见独立结构证据；独立性看不清则uncertain。merge必须shared，reject必须nonbuilding。bbox仅包括本体可见结构，不把遮挡树冠计入建筑外轮廓；推测遮挡地基单独放footprint_image。不要把原画source坐标移动到平面zone里，当前阶段根本没有平面布局坐标。
