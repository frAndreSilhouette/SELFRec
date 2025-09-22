import numpy as np
from collections import defaultdict
from data.data import Data
from data.graph import Graph
import scipy.sparse as sp


class Interaction(Data, Graph):
    def __init__(self, conf, training, test):
        Graph.__init__(self)
        Data.__init__(self, conf, training, test)

        self.user = {}
        self.item = {}
        self.id2user = {}
        self.id2item = {}
        self.training_set_u = defaultdict(dict)
        self.training_set_i = defaultdict(dict)
        self.test_set = defaultdict(dict)
        self.test_set_item = set()

        # self.current_itt = defaultdict(dict)
        # # 用于保存每个用户-商品最近一次购买至今的日期差，直接从txt文件读取，用于预测时embedding加权
        # # 格式：{"u1": {"i1": 45}}"}

        # self.weibull_params = defaultdict(dict)
        # # 用于保存每个用户-商品对应的 weibull 参数，直接从txt文件读取，用于预测时embedding加权
        # # 格式：{"i1":{"scale": 1.0, "shape": 1.0}}

        # 用矩阵/数组保存，不再使用dict of dict
        self.current_itt = None
        self.weibull_scale = None
        self.weibull_shape = None

        self.__generate_set()
        self.user_num = len(self.training_set_u)
        self.item_num = len(self.training_set_i)
        self.ui_adj = self.__create_sparse_bipartite_adjacency()
        self.norm_adj = self.normalize_graph_mat(self.ui_adj)
        self.interaction_mat = self.__create_sparse_interaction_matrix()



    def __generate_set(self):
        # 修改：先统计总用户数和总物品数
        num_users = len(set([x[0] for x in self.training_data]))
        num_items = len(set([x[1] for x in self.training_data]))
        # 修改：初始化weibull数组
        self.weibull_scale = np.zeros(num_items, dtype=np.float32)  # 【修改】
        self.weibull_shape = np.zeros(num_items, dtype=np.float32)  # 【修改】
        # 修改：暂存current_itt
        temp_current_itt = defaultdict(dict)  # 【修改】

        # 构造current_itt_cdf对应的numpy矩阵（缺失值填充-1，不需要计算均值，所以没有current_itt矩阵那么复杂）
        self.current_itt_cdf = -1 * np.ones((num_users, num_items), dtype=np.float32)

        for user, item, itt, scale, shape, rating, current_itt, current_itt_cdf in self.training_data: # 【这里修改】training_data现在包含了itt和分布信息
            if user not in self.user:
                user_id = len(self.user)
                self.user[user] = user_id
                self.id2user[user_id] = user
            if item not in self.item:
                item_id = len(self.item)
                self.item[item] = item_id
                self.id2item[item_id] = item # item是原来的字符串，item_id是对应的从0开始数字id

                # self.weibull_params[item]["scale"] = scale
                # self.weibull_params[item]["shape"] = shape
                # 修改：存入weibull数组
                self.weibull_scale[item_id] = scale  # 【修改】
                self.weibull_shape[item_id] = shape  # 【修改】
            
            self.training_set_u[user][item] = 1
            self.training_set_i[item][user] = 1

            self.current_itt_cdf[user_id, item_id] = current_itt_cdf

            # 样例：
            # self.training_set_u = {
            #     "u1": {"i1": 1},
            #     "u2": {"i1": 1, "i2": 1}
            # }
            # 哪怕training_data中有重复的user-item对，也只会存储一次，key对应的value为1             
            # self.current_itt[user][item] = current_itt

            # 修改：先存入临时dict
            temp_current_itt[user][item] = current_itt  # 【修改】

        # 修改：将current_itt转为numpy矩阵
        self.current_itt = np.zeros((len(self.user), len(self.item)), dtype=np.float32)  # 【修改】
        for user, items in temp_current_itt.items():  # 【修改】
            user_id = self.user[user]
            for item, itt in items.items():
                item_id = self.item[item]
                self.current_itt[user_id, item_id] = itt  # 【修改】

        # current_itt的空缺值用该item的均值填充
        for item_id in range(len(self.item)):
            # 找到该列非零值
            col_values = self.current_itt[:, item_id]
            non_zero_mask = col_values != 0
            if np.any(non_zero_mask):
                mean_val = col_values[non_zero_mask].mean()
                # 将0值替换为均值
                col_values[~non_zero_mask] = mean_val
                self.current_itt[:, item_id] = col_values
      
        for user, item, itt, scale, shape, rating, current_itt, current_itt_cdf in self.test_data: # 【这里修改】training_data现在包含了itt和分布信息
            if user in self.user and item in self.item:
                self.test_set[user][item] = 1
                self.test_set_item.add(item)

    def __create_sparse_bipartite_adjacency(self, self_connection=False):
        n_nodes = self.user_num + self.item_num
        user_np = np.array([self.user[pair[0]] for pair in self.training_data])
        item_np = np.array([self.item[pair[1]] for pair in self.training_data]) + self.user_num
        ratings = np.ones_like(user_np, dtype=np.float32)
        tmp_adj = sp.csr_matrix((ratings, (user_np, item_np)), shape=(n_nodes, n_nodes), dtype=np.float32)
        adj_mat = tmp_adj + tmp_adj.T
        if self_connection:
            adj_mat += sp.eye(n_nodes)
        return adj_mat

    def convert_to_laplacian_mat(self, adj_mat):
        user_np_keep, item_np_keep = adj_mat.nonzero()
        ratings_keep = adj_mat.data
        tmp_adj = sp.csr_matrix((ratings_keep, (user_np_keep, item_np_keep + adj_mat.shape[0])),
                                shape=(adj_mat.shape[0] + adj_mat.shape[1], adj_mat.shape[0] + adj_mat.shape[1]),
                                dtype=np.float32)
        tmp_adj = tmp_adj + tmp_adj.T
        return self.normalize_graph_mat(tmp_adj)

    def __create_sparse_interaction_matrix(self):
        row = np.array([self.user[pair[0]] for pair in self.training_data])
        col = np.array([self.item[pair[1]] for pair in self.training_data])
        entries = np.ones(len(row), dtype=np.float32)
        return sp.csr_matrix((entries, (row, col)), shape=(self.user_num, self.item_num), dtype=np.float32)

    def get_user_id(self, u):
        return self.user.get(u)

    def get_item_id(self, i):
        return self.item.get(i)
    # 将原始的用户或物品 ID（通常是字符串）映射到内部索引（数字，从 0 开始）

    def training_size(self):
        return len(self.user), len(self.item), len(self.training_data)

    def test_size(self):
        return len(self.test_set), len(self.test_set_item), len(self.test_data)

    def contain(self, u, i):
        return u in self.user and i in self.training_set_u[u]

    def contain_user(self, u):
        return u in self.user

    def contain_item(self, i):
        return i in self.item

    def user_rated(self, u):
        return list(self.training_set_u[u].keys()), list(self.training_set_u[u].values())

    def item_rated(self, i):
        return list(self.training_set_i[i].keys()), list(self.training_set_i[i].values())

    def row(self, u):
        k, v = self.user_rated(self.id2user[u])
        vec = np.zeros(self.item_num, dtype=np.float32)
        for item, rating in zip(k, v):
            vec[self.item[item]] = rating
        return vec

    def col(self, i):
        k, v = self.item_rated(self.id2item[i])
        vec = np.zeros(self.user_num, dtype=np.float32)
        for user, rating in zip(k, v):
            vec[self.user[user]] = rating
        return vec

    def matrix(self):
        m = np.zeros((self.user_num, self.item_num), dtype=np.float32)
        for u, u_id in self.user.items():
            vec = np.zeros(self.item_num, dtype=np.float32)
            k, v = self.user_rated(u)
            for item, rating in zip(k, v):
                vec[self.item[item]] = rating
            m[u_id] = vec
        return m
