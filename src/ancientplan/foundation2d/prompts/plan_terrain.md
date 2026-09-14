任务：把整幅古代山水画做成完整且可解释的二维俯视地基草图。此阶段处理地形、支承台地、树根与交通水域，建筑单独做下一阶段。第一张为全画；坐标以整张图x,y∈[0,1000]。图外装裱、印章、题字、天空、云和远山间留白不自动当水域/平地。

我们不能从散点透视古画恢复真实测量地图，要求一个空间关系合理的近似总图：保留左右/前后/相邻关系，高山在俯视中是占地块，不能直接把山峰高耸轮廓粘为底座。山上的房屋落在局部平地台地上，台地可以在山体大占地范围内，是叠置的支承层；树画树根位置/林地地表占地区，不把整个树冠遮挡面积当不可通行地基。不得添加凭空的支流、道路和建筑。

输出严格JSON：
{"stage":"plan_terrain","coordinate_system":"image_and_plan_1000",
 "source_boundary":[[x,y],...],"plan_boundary":[[x,y],...],
 "camera":{"kind":"multi_view|approx_parallel","elevation_degrees_range":[low,high],"basis":"可见侧墙和台地的证据；角度是估计不是标定","plan_convention":"图上为画面远处，非真北；平面相对尺度，非米"},
 "terrain":[{"id":"T01","category":"mountain|water|flat|other","label":"短名",
 "source_polygon":[[x,y],...],"plan_polygon":[[x,y],...],
 "level":"low|middle|high","confidence":"high|medium|low","basis":"哪些是可见，哪些是底座/遮挡补全假设"}],
 "trees":[{"id":"V01","label":"短名","source_roots":[[x,y],...],"plan_roots":[[x,y],...],"grouped":true,"confidence":"high|medium|low","basis":"根部证据，若一片密林是代表性根点，不是假装每株精确识别"}],
 "routes":[{"id":"P01","kind":"path|bridge|steps","source_points":[[x,y],...],"plan_points":[[x,y],...],"confidence":"high|medium|low","basis":"画中路径证据，无法追踪的断开而非连出虚假线路"}],
 "zones":[{"id":"Z01","label":"建筑群/台地名称","source_polygon":[[x,y],...],"plan_polygon":[[x,y],...],"level":"low|middle|high","ground_axes_image":[[dx,dy],[dx,dy]],"basis":"该处地面两方向；仅用作推测网格，不是严格相机标定"}],
 "unknown_areas":[{"source_polygon":[...],"reason":"不确定是什么地面/远景/留白"}],
 "assumptions":["整体映射/压缩高差的方法和不确定性"]}

plan_polygon/plan_roots等为你直接提出的共享1000x1000近似俯视坐标，不是把原画每个点按相同比例抄过去。根据山脚、层次和连通关系压缩竖向山体高度、适度拉开前后被遮挡的地面，支承台地按合理比例放置。保留总体画面左右排列，无需强行单灭点。source_polygon记录原画对应可见对象/地表，允许与补全底座不同。
地形范围覆盖整幅的主要前/中/远景，别只做局部。每个建筑集中区都需要zones，供后续把地基放在台地上；可以有坡上小台地，不能把山整体当平地。少量顶层连通山体可合并，但不同岸线/高差必须分开。有水只标有明确岸线/船/水纹证据的范围。多边形简单不自交，单个最多18点。树根每组最多12个代表点，可多组覆盖整画。

绘制任何山/树/建筑支承面前先排查水中倒影：纸绢色区域不一定是陆地，蓝绿色区域也不一定是实体山。用正立/倒置结构、与岸线上方轮廓的对应、连续岸线和水纹共同判断。确认是反射像的山、树、房屋，不再生成第二份山体占地、树根或建筑台地；其承载面只能依据水域证据标注，证据不足则明确未定，不能把未知水岸硬说成平地。
