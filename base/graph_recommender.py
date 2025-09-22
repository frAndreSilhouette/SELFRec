from base.recommender import Recommender
from data.ui_graph import Interaction
from util.algorithm import find_k_largest
from time import strftime, localtime, time
from data.loader import FileIO
from os.path import abspath
from util.evaluation import ranking_evaluation
from datetime import datetime
from tqdm import tqdm


class GraphRecommender(Recommender):
    def __init__(self, conf, training_set, test_set, **kwargs):
        super(GraphRecommender, self).__init__(conf, training_set, test_set, **kwargs)
        self.data = Interaction(conf, training_set, test_set)
        self.bestPerformance = []
        self.topN = [int(num) for num in self.ranking]
        # self.max_N = max(self.topN) # 选最优模型是按照最大的top N来看的
        self.max_N = 10 # 选最优模型是按照top 10来看的

    def print_model_info(self, loss_func): # 【这里修改】增加了loss_func参数
        super(GraphRecommender, self).print_model_info(loss_func)
        # print dataset statistics
        print(f'Training Set Size: (user number: {self.data.training_size()[0]}, '
              f'item number: {self.data.training_size()[1]}, '
              f'interaction number: {self.data.training_size()[2]})')
        print(f'Test Set Size: (user number: {self.data.test_size()[0]}, '
              f'item number: {self.data.test_size()[1]}, '
              f'interaction number: {self.data.test_size()[2]})')
        print('=' * 80)

    def build(self):
        pass

    def train(self): # 【这里修改】train方法不再需要loss_func参数
        pass

    def predict(self, u):
        pass
    
    # 原版本
    def test(self):
        # def process_bar(num, total):
        #     rate = float(num) / total
        #     ratenum = int(50 * rate)
        #     print(f'\rProgress: [{"+" * ratenum}{" " * (50 - ratenum)}]{ratenum * 2}%', end='', flush=True)

        rec_list = {}
        user_count = len(self.data.test_set)
        # for i, user in enumerate(self.data.test_set):
        for i, user in tqdm(enumerate(self.data.test_set), desc="Processing users"):
            candidates = self.predict(user)
            rated_list, _ = self.data.user_rated(user)
            # for item in rated_list:
            #     candidates[self.data.item[item]] = -10e8  # 将买过的商品打分设为很小，防止被推荐【我不要这个逻辑】
            ids, scores = find_k_largest(self.max_N, candidates)
            item_names = [self.data.id2item[iid] for iid in ids]
            rec_list[user] = list(zip(item_names, scores))
            # if i % 1000 == 0:
            #     process_bar(i, user_count)
        # process_bar(user_count, user_count)
        # print('')
        return rec_list

    # # 新版本：加上hard的召回插件，process_bar更改为tqdm
    # def test(self):
    #     rec_list = {}
    #     user_ids = list(self.data.test_set.keys()WWW)

    #     for user in tqdm(user_ids, desc="Processing users"):
    #         candidates = self.predict(user)
    #         rated_list, _ = self.data.user_rated(user)

    #         # ---------- 原有逻辑：选取 top-k ----------
    #         ids, scores = find_k_largest(self.max_N, candidates)
    #         item_names = [self.data.id2item[iid] for iid in ids]
    #         original_rec = list(zip(item_names, scores))

    #         # ---------- 判断是否需要生成补充清单 ----------
    #         if self.quantile_bpr_loss.loss_func != 0:
    #             # print('[DEBUG] loss func非0，使用hard rule')
    #             retrieval_list = []
    #             user_id = self.data.user[user]  # 用户索引
    #             for item_id in range(self.data.current_itt_cdf.shape[1]):
    #                 cdf_val = self.data.current_itt_cdf[user_id, item_id]
    #                 if cdf_val == -1:
    #                     continue  # 用户未购买过
    #                 shape = self.data.weibull_shape[item_id]
    #                 threshold = 0.3 if shape >= 1 else 0.7
    #                 if cdf_val > threshold:
    #                     retrieval_list.append((self.data.id2item[item_id], -1))  # score 统一设置为 -1

    #             # 按 CDF 降序排列
    #             retrieval_list.sort(key=lambda x: self.data.current_itt_cdf[user_id, self.data.item[x[0]]], reverse=True)

    #             # ---------- 合并清单并去重 ----------
    #             seen = set()
    #             final_list = []
    #             for item, score in retrieval_list + original_rec:
    #                 if item not in seen:
    #                     final_list.append((item, score))
    #                     seen.add(item)

    #             rec_list[user] = final_list
    #         else:
    #             # 直接返回原清单
    #             # print('[DEBUG] loss func为0，使用原推荐清单')
    #             rec_list[user] = original_rec

    #     return rec_list

    def evaluate(self, rec_list, loss_func):
        self.recOutput.append('userId: recommendations in (itemId, ranking score) pairs, * means the item is hit.\n')
        for user in self.data.test_set:
            line = user + ':' + ''.join(
                f" ({item[0]},{item[1]}){'*' if item[0] in self.data.test_set[user] else ''}"
                for item in rec_list[user]
            )
            line += '\n'
            self.recOutput.append(line)
        # current_time = strftime("%Y-%m-%d %H-%M-%S", localtime(time()))
        current_time = datetime.now().strftime("%Y-%m-%d %H-%M-%S-%f") # 精确到微秒
        out_dir = self.output
        # file_name = f"{self.config['model']['name']}_loss{loss_func}@{current_time}-top-{self.max_N}items.txt"
        # FileIO.write_file(out_dir, file_name, self.recOutput)
        # print('The result has been output to ', abspath(out_dir), '.') # 【这里修改】暂时去除推荐清单输出
        file_name = f"{self.config['model']['name']}_loss{loss_func}@{current_time}-performance.txt"
        self.result = ranking_evaluation(self.data.test_set, rec_list, self.topN)
        self.model_log.add('###Evaluation Results###')
        self.model_log.add(self.result)
        FileIO.write_file(out_dir, file_name, self.result)
        print(f'The result of {self.model_name}:\n{"".join(self.result)}')

    def fast_evaluation(self, epoch):
        print('Evaluating the model...')
        rec_list = self.test()
        measure = ranking_evaluation(self.data.test_set, rec_list, [self.max_N])

        performance = {k: float(v) for m in measure[1:] for k, v in [m.strip().split(':')]}

        # 评价标准：某指标更高+1，更低-1，看分数是否>0
        if self.bestPerformance:
            count = sum(1 if self.bestPerformance[1][k] > performance[k] else -1 for k in performance)
            if count < 0:
                self.bestPerformance = [epoch + 1, performance]
                self.save()
        else:
            self.bestPerformance = [epoch + 1, performance]
            self.save()

        # # 评价标准：Top 10 Recall
        # if self.bestPerformance:
        #     if performance["Recall"] > self.bestPerformance[1]["Recall"]:
        #         self.bestPerformance = [epoch + 1, performance]
        #         self.save()
        # else:
        #     self.bestPerformance = [epoch + 1, performance]
        #     self.save()

        print('-' * 80)
        print(f'Real-Time Ranking Performance (Top-{self.max_N} Item Recommendation)')
        measure_str = ', '.join([f'{k}: {v}' for k, v in performance.items()])
        print(f'*Current Performance*\nEpoch: {epoch + 1}, {measure_str}')
        bp = ', '.join([f'{k}: {v}' for k, v in self.bestPerformance[1].items()])
        print(f'*Best Performance*\nEpoch: {self.bestPerformance[0]}, {bp}')
        print('-' * 80)
        return measure
