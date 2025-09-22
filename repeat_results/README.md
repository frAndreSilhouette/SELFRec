现在是只考虑embedding加权，不考虑item异质性，单校区重复10次

后台跑两个main.py，每个重复5次实验

v1 每个epoch结束后都evaluate，很耗时间

v2 10个epoch结束后再evaluate

v3 文件名精确到毫秒，避免重复覆盖；完整的10次重复实验结果，有使用价值

v4 未购买商品也纳入推荐，使用其他人的current_itt均值进行加权（而不是权重设置为0）；按top 10的四个指标选最优模型，有使用价值

v5 在v4的基础上，仅考虑loss0和loss3；使用2024-12-18及之后的数据作为测试集（2周）（未完成）
campus 10: train 386207 行, test 32560 行
campus 15: train 740660 行, test 62183 行
campus 34: train 206948 行, test 15189 行
campus 143: train 112678 行, test 9545 行
campus 102: train 342129 行, test 27561 行

v6 仅考虑loss0和loss3，两周测试集，loss3加上hard召回插件（非耐用品0.3，耐用品0.7）；考虑硬召回的协同过滤（CF）