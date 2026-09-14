根据原始画作审查随附的结构候选 JSON。数据中的线都是上一轮模型提出的，不是图像证据；必须回看原画。
本次不修改坐标，只判断哪些对象的结构/地基适合进入下一步，哪些必须搁置复核。

重点：可见线是否落在真实脊檐墙柱/地形边线；把房子拆重复了吗；隐藏线是否有参照；屋顶或上层栏杆是否被当成地基；山峰/树冠是否被当成山脚/树根；不知的闭合边是否被谎标 visible；补全是否穿过明显通道或同层其他建筑。
每个输入 ID 必须恰好一个判断。不要只检查房屋而跳过地形。
输出紧凑 JSON：
{
 "stage":"structure_audit",
 "objects":[{"id":"沿用输入ID","verdict":"needs_review","issues":["明确问题或不确定点"],"safe_for_ground_projection":false}],
 "summary":"指出主要错误与仍未验证之处"
}
verdict 只能 plausible/needs_review/reject。plausible 仅表示这轮模型未发现明显矛盾，不是人工核验；只有 plausible 且地面接触确有依据，safe_for_ground_projection 才能 true。
若上一轮 status=deferred 或 footprint.kind=unknown，safe_for_ground_projection 必须 false。
不得把猜测说成真值，不输出思考过程。
