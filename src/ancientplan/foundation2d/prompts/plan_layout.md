你正在完成整幅古画的统一俯视地基总图，不是独立矩形清单。第1张是原画，其他图为辅助。输入JSON是各局部独立识别候选和地形方案（不是已确认真值）。所有输入坐标已映射为原始整图0..1000。

工作：1逐个核对建筑候选并跨裁片去重；连续房屋的屋顶/墙体/上下层不能重复，独立同院房屋不能误合；2将保留建筑放到给定地形方案的共享plan坐标和正确台地；3选择合理的前墙宽/进深、朝向、遮挡补全，并修正互相重叠/堵路/落水/落悬崖问题；4扫视全画找真正遗漏实例，若新发现必须单独给出证据，不凑固定数量。不要把图像bbox宽高比例直接当占地比例。

输出JSON：
{"stage":"plan_layout","buildings":[
 {"id":"H01","source_ids":["nw:B01",...],"category":"house|other","subtype":"...","label":"短名",
 "source_bbox":[x0,y0,x1,y1],"source_anchor":[x,y],
 "source_footprint":[[x,y],[x,y],[x,y],[x,y]],
 "zone_id":"Zxx","plan_center":[x,y],"plan_size":[前墙宽,进深],
 "front_clock":6,"ratio_range":[low,high],"confidence":"high|medium|low",
 "evidence":"身份和支承关系证据","assumption":"隐藏边/比例/位置的假设"}],
 "rejected":[{"source_id":"nw:Bxx","reason":"具体证据"}],
 "relations":[{"from":"Hxx","to":"Hxx或Zxx","relation":"same_compound|in_front_of|behind|left_of|access","basis":"依据"}],
 "coverage_review":[{"region":"左/中/右 前/中/远景具体部位","status":"covered|uncertain","note":"检查了什么，缺失若无证据请说明"}],
 "assumptions":["共享地图比例与多个局部视角的近似"]}

每个输入candidate.id必须恰好出现在一个building.source_ids或rejected.source_id里。不同裁片重复同一栋合并source_ids。新增实例source_ids=[]并evidence写newly_observed以及可见证据。保留实例在source里是整图0..1000坐标。source_footprint必须凸四边形，按环绕顺序；全部视为假设，不假装实测墙脚。
去重合并不等于拒绝：两个候选都并入一栋时，把两个ID只写入该栋source_ids，绝不能再在rejected里写“已合并/仅作去重说明”。rejected只记录没有分配给任何地基的候选。输出前逐一核对全部输入ID，既不能遗漏，也不能同时保留和拒绝。
永久亭阁/敞轩具有独立地基或固定桩基时归house并保留subtype；没有封闭墙不是other的充分理由。临时棚、桥、连廊和围挡按其自身支承证据归other。
plan_center/plan_size为1000共享平面单位，非米。front_clock为建筑正面朝向：12向上、3向右、6向下、9向左；plan_size[0]始终是正立面宽度，第二项为进深。渲染器自动把四角旋转，不要用屋脊方向代替正面方向。两层屋的plan_size不得因层数而加倍。大殿、长条厢房、塔、亭应有各自合理比例，平面矩形允许基本假设；连廊可以长条，独立房屋不要拉成长板。
尽量把建筑落在同名支承zone.plan_polygon范围内，允许极小误差；不得为了装下错误合并而把房子扩大到整院。多建筑院落保留院心空地。不要改动输入terrain；若明显矛盾写assumptions待下一轮统一修正。

重要：候选中的front_clock可能只是上游忽略了方向而填6，绝不能机械继承！本次你必须用图像重新判断每栋的正面（入口/长墙柱列）与侧墙关系。模板的6仅是类型示例，不是默认答案。若看见一侧墙朝画面左下，一侧朝右下，要区分哪一面为正面；沿屋脊看的方向不是房子朝向。给出的clock是俯视近似下的朝向，不是图像边线斜率；允许5、6、7、8等多种，证据不足在assumption里说明。
