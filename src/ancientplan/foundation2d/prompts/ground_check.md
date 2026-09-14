对照原画审查地基假设。输入包括 landmarks（逐点线索）和 hypotheses（推测地基），它们都是模型输出，不是图像真值。
所有坐标相对图1；如果有图2叠加图，只帮助找到候选位置，其线条不是额外图像证据。

本次只审查、不修改坐标。逐个 alternative 给出 verdict：
- usable_hypothesis：可作为明确标注不确定性的俯视候选比较，不表示测量准确或真实恢复。
- needs_revision：位置/比例/支承面存在明显未解决问题，不能进入默认地基图。
- reject：明显违反可见结构（如窗槛作墙脚、地基穿出侧墙、同一房子重复占地等）。

要检查：真实 ground 点是否正确；是否把遮挡交点提升成确定地基；长宽比是否只是图像框比例；阁楼地基是否落在中间楼层；推测深度有无自洽参考；同层建筑地基是否互相穿插。
特别检查四角顺序和宽/深是否一致：A→B 必须沿主要门窗所在的前墙，B→C 才是进深。若 A、B 实际是同一短侧墙两端，却在 ratio_basis 中把另一条长边说成面宽，应判 needs_revision 并指出要怎样改对应关系。
“点是ground”也要回看原图，不盲信 landmarks：檐柱脚可能只是楼上露台的柱脚，台基中部转角也可能没有到最低接地层。
完全未知的后边可作为明确的假设，不因含 inferred 就一律拒绝；但不能用“是假设”回避与可见墙身或门前通道的矛盾。
shape_only 不具有位置，不要评价其在画面的落点。

输出 JSON：
{"stage":"ground_hypothesis_check","decisions":[{"instance_id":"G1","alternative_id":"P1","evaluated_corners":[],"front_edge_indices":[0,1],"order_consistent":true,"verdict":"needs_revision","reason":"具体支持或矛盾","missing_evidence":"哪些还未验证"}],"identity_warnings":["仍有实例混淆之处"],"summary":"是否足够制作带假设标签的局部俯视候选，哪些不能"}
evaluated_corners 必须逐点原样复制你正在评估的 image_corners，保持原有顺序，不在解释里偷偷换字母。
front_edge_indices 是实际主要前墙在该数组中的有向顶点索引（0起）：若应是1→2，就填[1,2]并 order_consistent=false、verdict=needs_revision，不能声称A→B一致。完全无法判断前墙则 []、order_consistent=false，不能 usable_hypothesis。
shape_only 对象 evaluated_corners=[]、front_edge_indices=[]，order_consistent=true（不涉及落点四角次序）。
每个原 alternative 恰好一个判断；没有候选的实例不用补造。不要输出思考过程。
