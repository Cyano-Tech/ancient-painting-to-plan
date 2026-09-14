仅修复上一阶段的源ID归属元数据，不得改变建筑数量、尺寸、坐标或朝向。
input.expected_source_ids里的每个ID必须恰好出现一次。input.layout.buildings是已经完成合并/删除后的实际实例数组。input.layout.rejected可能含模型文字说“清空/不应在此/已并入某栋”但实际没删掉的条目。
对于已明确合并进保留实例source_ids的ID，把target指定为该保留实例ID；如果确实属于未保留的误检，则target=null并说明原因。
输出严格JSON：{"stage":"plan_accounting","assignments":[{"source_id":"nw:B01","target":"H01","reason":"已合并到该实例"},...]}
target只能为input.layout.buildings里的已有ID或null。每个源ID仅一条，不要又在另一个数组重复解释。需要空数组时必须真的输出[]，不能在数组里写“此项应删除”。几何识别结果不能在本阶段改动。
