import math


class Metric(object):
    def __init__(self):
        pass

    @staticmethod
    def hits(origin, res):
        hit_count = {}
        for user in origin:
            items = list(origin[user].keys())
            predicted = [item[0] for item in res[user]]
            hit_count[user] = len(set(items).intersection(set(predicted)))
        return hit_count

    @staticmethod
    def hit_ratio(origin, hits):
        """
        Note: This type of hit ratio calculates the fraction:
         (# retrieved interactions in the test set / #all the interactions in the test set)
        """
        total_num = 0
        for user in origin:
            items = list(origin[user].keys())
            total_num += len(items)
        hit_num = 0
        for user in hits:
            hit_num += hits[user]
        return round(hit_num/total_num,5)

    # # @staticmethod
    # def hit_ratio(origin, hits):
    #     """
    #     Note: This type of hit ratio calculates the fraction:
    #      (# users who are recommended items in the test set / #all the users in the test set)
    #     """
    #     hit_num = 0
    #     for user in hits:
    #         if hits[user] > 0:
    #             hit_num += 1
    #     return hit_num / len(origin)

    @staticmethod
    def precision(hits, N):
        prec = sum([hits[user] for user in hits])
        return round(prec / (len(hits) * N),5)

    @staticmethod
    def recall(hits, origin):
        recall_list = [hits[user]/len(origin[user]) for user in hits]
        recall = round(sum(recall_list) / len(recall_list),5)
        return recall

    @staticmethod
    def F1(prec, recall):
        if (prec + recall) != 0:
            return round(2 * prec * recall / (prec + recall),5)
        else:
            return 0

    @staticmethod
    def MAE(res):
        error = 0
        count = 0
        for entry in res:
            error+=abs(entry[2]-entry[3])
            count+=1
        if count==0:
            return error
        return round(error/count,5)

    @staticmethod
    def RMSE(res):
        error = 0
        count = 0
        for entry in res:
            error += (entry[2] - entry[3])**2
            count += 1
        if count==0:
            return error
        return round(math.sqrt(error/count),5)

    @staticmethod
    def NDCG(origin,res,N):
        sum_NDCG = 0
        for user in res:
            DCG = 0
            IDCG = 0
            #1 = related, 0 = unrelated
            for n, item in enumerate(res[user]):
                if item[0] in origin[user]:
                    DCG+= 1.0/math.log(n+2,2)
            for n, item in enumerate(list(origin[user].keys())[:N]):
                IDCG+=1.0/math.log(n+2,2)
            sum_NDCG += DCG / IDCG
        return round(sum_NDCG / len(res),5)

    @staticmethod
    def MAP(origin, res, N):
        sum_prec = 0
        for user in res:
            hits = 0
            precision = 0
            for n, item in enumerate(res[user]):
                if item[0] in origin[user]:
                    hits += 1
                    precision += hits / (n + 1.0)
            sum_prec += precision / min(len(origin[user]), N)
        return sum_prec / len(res)

    @staticmethod
    def AUC(origin, res, rawRes):
    
        from random import choice
        sum_AUC = 0
        for user in origin:
            count = 0
            larger = 0
            itemList = rawRes[user].keys()
            for item in origin[user]:
                item2 = choice(itemList)
                count += 1
                try:
                    if rawRes[user][item] > rawRes[user][item2]:
                        larger += 1
                except KeyError:
                    count -= 1
            if count:
                sum_AUC += float(larger) / count
    
        return float(sum_AUC) / len(origin)

# # =============================
# # 这是GPT改写后的版本
# import math
# from random import choice

# class Metric(object):
#     def __init__(self):
#         pass

#     @staticmethod
#     def hits(origin, res):
#         """
#         统计每个用户 Top-K 推荐中命中的相关物品数量
#         origin: dict[user] = {item: rating, ...}
#         res: dict[user] = [(item, score), ...]
#         """
#         hit_count = {}
#         for user in origin:
#             true_items = set(origin[user].keys())
#             predicted = [item for item, _ in res[user]]
#             hit_count[user] = len(true_items.intersection(set(predicted)))
#         return hit_count

#     @staticmethod
#     def hit_ratio(origin, hits):
#         """
#         Hit Ratio (user-level): 至少命中一个相关物品算1，否则0，再取平均
#         """
#         hit_users = sum(1 for user in hits if hits[user] > 0)
#         return round(hit_users / len(origin), 5)

#     @staticmethod
#     def precision(hits, N):
#         """
#         Precision@K (macro average): 每个用户的 precision@K，最后取平均
#         """
#         precisions = []
#         for user in hits:
#             precisions.append(hits[user] / N)
#         return round(sum(precisions) / len(precisions), 5)

#     @staticmethod
#     def recall(hits, origin):
#         """
#         Recall@K (macro average): 每个用户的 recall@K，最后取平均
#         """
#         recalls = []
#         for user in hits:
#             if len(origin[user]) > 0:
#                 recalls.append(hits[user] / len(origin[user]))
#         return round(sum(recalls) / len(recalls), 5)

#     @staticmethod
#     def F1(prec, recall):
#         if (prec + recall) > 0:
#             return round(2 * prec * recall / (prec + recall), 5)
#         else:
#             return 0.0

#     @staticmethod
#     def MAE(res):
#         """
#         res: list of (user, item, true_rating, predicted_rating)
#         """
#         error, count = 0, 0
#         for entry in res:
#             error += abs(entry[2] - entry[3])
#             count += 1
#         return round(error / count, 5) if count > 0 else 0.0

#     @staticmethod
#     def RMSE(res):
#         """
#         res: list of (user, item, true_rating, predicted_rating)
#         """
#         error, count = 0, 0
#         for entry in res:
#             error += (entry[2] - entry[3]) ** 2
#             count += 1
#         return round(math.sqrt(error / count), 5) if count > 0 else 0.0

#     @staticmethod
#     def NDCG(origin, res, N):
#         """
#         Normalized Discounted Cumulative Gain@K
#         """
#         total_ndcg = 0
#         for user in res:
#             DCG, IDCG = 0.0, 0.0
#             # DCG
#             for rank, (item, _) in enumerate(res[user][:N]):
#                 if item in origin[user]:
#                     DCG += 1.0 / math.log(rank + 2, 2)
#             # IDCG (取用户真实相关物品数和K的最小值)
#             max_rel = min(len(origin[user]), N)
#             for rank in range(max_rel):
#                 IDCG += 1.0 / math.log(rank + 2, 2)
#             if IDCG > 0:
#                 total_ndcg += DCG / IDCG
#         return round(total_ndcg / len(res), 5)

#     @staticmethod
#     def MAP(origin, res, N):
#         """
#         Mean Average Precision@K
#         """
#         sum_ap = 0
#         for user in res:
#             hits, ap = 0, 0.0
#             true_items = set(origin[user].keys())
#             for rank, (item, _) in enumerate(res[user][:N]):
#                 if item in true_items:
#                     hits += 1
#                     ap += hits / (rank + 1.0)
#             if len(true_items) > 0:
#                 sum_ap += ap / min(len(true_items), N)
#         return round(sum_ap / len(res), 5)

#     @staticmethod
#     def AUC(origin, res, rawRes):
#         """
#         AUC: 计算正负样本对的排序概率
#         origin: dict[user] = {positive_item: rating, ...}
#         res: dict[user] = [(item, score), ...]  (排序后的预测)
#         rawRes: dict[user] = {item: score, ...} (未排序预测分数)
#         """
#         total_auc = 0
#         for user in origin:
#             pos_items = set(origin[user].keys())
#             all_items = set(rawRes[user].keys())
#             neg_items = all_items - pos_items
#             if not neg_items or not pos_items:
#                 continue
#             pair_count, correct = 0, 0
#             for pos in pos_items:
#                 for neg in neg_items:
#                     pair_count += 1
#                     if rawRes[user].get(pos, 0) > rawRes[user].get(neg, 0):
#                         correct += 1
#             if pair_count > 0:
#                 total_auc += correct / pair_count
#         return round(total_auc / len(origin), 5)


def ranking_evaluation(origin, res, N):
    measure = []
    for n in N:
        predicted = {}
        for user in res:
            predicted[user] = res[user][:n]
        indicators = []
        if len(origin) != len(predicted):
            print('The Lengths of test set and predicted set do not match!')
            exit(-1)
        hits = Metric.hits(origin, predicted)
        hr = Metric.hit_ratio(origin, hits)
        indicators.append('Hit Ratio:' + str(hr) + '\n')
        prec = Metric.precision(hits, n)
        indicators.append('Precision:' + str(prec) + '\n')
        recall = Metric.recall(hits, origin)
        indicators.append('Recall:' + str(recall) + '\n')
        # F1 = Metric.F1(prec, recall)
        # indicators.append('F1:' + str(F1) + '\n')
        # MAP = Metric.MAP(origin, predicted, n)
        # indicators.append('MAP:' + str(MAP) + '\n')
        NDCG = Metric.NDCG(origin, predicted, n)
        indicators.append('NDCG:' + str(NDCG) + '\n')
        # AUC = Metric.AUC(origin,res,rawRes)
        # measure.append('AUC:' + str(AUC) + '\n')
        measure.append('Top ' + str(n) + '\n')
        measure += indicators
    return measure

def rating_evaluation(res):
    measure = []
    mae = Metric.MAE(res)
    measure.append('MAE:' + str(mae) + '\n')
    rmse = Metric.RMSE(res)
    measure.append('RMSE:' + str(rmse) + '\n')
    return measure