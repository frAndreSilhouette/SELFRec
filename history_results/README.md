v1: 仅将12月当成test set
campus 10: train 340723 行, test 73362 行
campus 15: train 661738 行, test 132735 行
campus 34: train 180069 行, test 38649 行
campus 143: train 100240 行, test 20913 行
campus 102: train 305466 行, test 60335 行

v2：将11-12月当成test set
campus 10: train 270779 行, test 141591 行
campus 15: train 534342 行, test 255291 行
campus 34: train 140086 行, test 78441 行
campus 143: train 79754 行, test 40298 行
campus 102: train 239870 行, test 123061 行

v3: 将11-12月当成test set，去除“不推荐已购买商品”的逻辑
campus 10: train 270779 行, test 141591 行, test占比 34.34%
campus 15: train 534342 行, test 255291 行, test占比 32.37%
campus 34: train 140086 行, test 78441 行, test占比 35.96%
campus 143: train 79754 行, test 40298 行, test占比 33.54%
campus 102: train 239870 行, test 123061 行, test占比 33.99%

v4: 将最后一周当成test set，将bpr损失改在init中，只考虑campus143，预测embedding乘5种权重（新predict函数）
campus 10: train 402244 行, test 16523 行
campus 15: train 772578 行, test 30265 行
campus 34: train 216942 行, test 5195 行
campus 143: train 117392 行, test 4831 行
campus 102: train 355634 行, test 14056 行

v5: 将最后一周当成test set，将bpr损失改在init中，只考虑campus143，预测embedding不乘权重

v6: 将最后一周当成test set，将bpr损失改在init中，只考虑campus15，预测embedding乘5种权重（新predict函数）（只是为了不让机器晚上闲着）

v7: 将最后一周当成test set，将bpr损失改在init中，预测embedding乘5种权重（新predict函数），负样本预测权重为1e-6（防止都为0）

v8: 将最后一周当成test set，将bpr损失改在init中，预测embedding乘5种权重（新predict函数），负样本预测权重为0（和v4，v6同样的处理方式，只是将单校区改成了全校区）

v9: 在v4/6/8的基础上，score在embedding内积的基础上，加入sigmoid(w_1 * scale + w_2 * shape + b)表示商品异质性，五个校区

v10：在v4/6/8的基础上，score在embedding内积的基础上，加入w_1 * scale + w_2 * shape + b表示商品异质性，五个校区；修复了没保存quantile_bpr_loss最优参数的bug