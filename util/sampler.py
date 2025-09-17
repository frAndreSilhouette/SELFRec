from random import shuffle,randint,choice,sample
import numpy as np
import pandas as pd

def next_batch_pairwise(data,batch_size,n_negs=1):
    training_data = data.training_data
    # training_data实际上就是txt文件直接读取的结果，包含复购，所以
    # print(f"[DEBUG] training_data:{pd.DataFrame(training_data).shape}")
    shuffle(training_data)
    ptr = 0
    data_size = len(training_data)
    while ptr < data_size:
        if ptr + batch_size < data_size:
            batch_end = ptr + batch_size
        else:
            batch_end = data_size
        users = [training_data[idx][0] for idx in range(ptr, batch_end)]
        items = [training_data[idx][1] for idx in range(ptr, batch_end)]
        itts = [training_data[idx][2] for idx in range(ptr, batch_end)] # itt是距离上次购买的天数(int)
        pos_scales = [training_data[idx][3] for idx in range(ptr, batch_end)] # pos_scale是购买过的正样本的weibull分布的scale参数
        pos_shapes = [training_data[idx][4] for idx in range(ptr, batch_end)] # pos_shape是购买过的正样本的weibull分布的shape参数
        ptr = batch_end

        user_idx, pos_idx, neg_idx = [], [], []
        neg_scales, neg_shapes = [], []

        item_list = list(data.item.keys()) # item_list是一个列表，包含了所有的item id
        for i, user in enumerate(users): #遍历当前batch的所有user
            pos_idx.append(data.item[items[i]])
            user_idx.append(data.user[user])
            for m in range(n_negs):
                neg_item = choice(item_list)
                while neg_item in data.training_set_u[user]: # 检查neg_item是否在data.training_set_u[user]这个字典的键中（即是否购买过）
                    neg_item = choice(item_list)
                temp_neg_idx = data.item[neg_item]
                # neg_idx.append(data.item[neg_item])
                # neg_scales.append(data.weibull_params[neg_item]["scale"])
                # neg_shapes.append(data.weibull_params[neg_item]["shape"])
                
                # weibull_params拆成了两个numpy array
                neg_idx.append(temp_neg_idx)
                neg_scales.append(data.weibull_scale[temp_neg_idx])
                neg_shapes.append(data.weibull_shape[temp_neg_idx])

        yield user_idx, pos_idx, neg_idx, itts, pos_scales, pos_shapes, neg_scales, neg_shapes
        # 返回的user_idx和pos_idx实际上就是training_data的batch_size那么多行，neg_idx是抽的别的没买过的负样本



def next_batch_pointwise(data,batch_size):
    training_data = data.training_data
    data_size = len(training_data)
    ptr = 0
    while ptr < data_size:
        if ptr + batch_size < data_size:
            batch_end = ptr + batch_size
        else:
            batch_end = data_size
        users = [training_data[idx][0] for idx in range(ptr, batch_end)]
        items = [training_data[idx][1] for idx in range(ptr, batch_end)]
        ptr = batch_end
        user_idx, pos_idx, y = [], [], []
        for i, user in enumerate(users):
            pos_idx.append(data.item[items[i]])
            user_idx.append(data.user[user])
            y.append(1)
            for instance in range(4):
                item_j = randint(0, data.item_num - 1)
                while data.id2item[item_j] in data.training_set_u[user]:
                    item_j = randint(0, data.item_num - 1)
                user_idx.append(data.user[user])
                pos_idx.append(item_j)
                y.append(0)
        yield user_idx, pos_idx, y

# def next_batch_sequence(data, batch_size,n_negs=1):
#     training_data = data.training_set
#     shuffle(training_data)
#     ptr = 0
#     data_size = len(training_data)
#     item_list = list(range(1,data.item_num+1))
#     while ptr < data_size:
#         if ptr+batch_size<data_size:
#             end = ptr+batch_size
#         else:
#             end = data_size
#         seq_len = []
#         batch_max_len = max([len(s[0]) for s in training_data[ptr: end]])
#         seq = np.zeros((end-ptr, batch_max_len),dtype=int)
#         pos = np.zeros((end-ptr, batch_max_len),dtype=int)
#         y = np.zeros((1, end-ptr),dtype=int)
#         neg = np.zeros((1,n_negs, end-ptr),dtype=int)
#         for n in range(0, end-ptr):
#             seq[n, :len(training_data[ptr + n][0])] = training_data[ptr + n][0]
#             pos[n, :len(training_data[ptr + n][0])] = list(reversed(range(1,len(training_data[ptr + n][0])+1)))
#             seq_len.append(len(training_data[ptr + n][0]) - 1)
#         y[0,:]=[s[1] for s in training_data[ptr:end]]
#         for k in range(n_negs):
#             neg[0,k,:]=sample(item_list,end-ptr)
#         ptr=end
#         yield seq, pos, seq_len, y, neg

def next_batch_sequence(data, batch_size,n_negs=1,max_len=50):
    training_data = [item[1] for item in data.original_seq]
    shuffle(training_data)
    ptr = 0
    data_size = len(training_data)
    item_list = list(range(1,data.item_num+1))
    while ptr < data_size:
        if ptr+batch_size<data_size:
            batch_end = ptr+batch_size
        else:
            batch_end = data_size
        seq = np.zeros((batch_end-ptr, max_len),dtype=int)
        pos = np.zeros((batch_end-ptr, max_len),dtype=int)
        y =np.zeros((batch_end-ptr, max_len),dtype=int)
        neg = np.zeros((batch_end-ptr, max_len),dtype=int)
        seq_len = []
        for n in range(0, batch_end-ptr):
            start = len(training_data[ptr + n]) > max_len and -max_len or 0
            end =  len(training_data[ptr + n]) > max_len and max_len-1 or len(training_data[ptr + n])-1
            seq[n, :end] = training_data[ptr + n][start:-1]
            seq_len.append(end)
            pos[n, :end] = list(range(1,end+1))
            y[n, :end]=training_data[ptr + n][start+1:]
            negatives=sample(item_list,end)
            while len(set(negatives).intersection(set(training_data[ptr + n][start:-1]))) >0:
                negatives = sample(item_list, end)
            neg[n,:end]=negatives
        ptr=batch_end
        yield seq, pos, y, neg, np.array(seq_len,int)

def next_batch_sequence_for_test(data, batch_size,max_len=50):
    sequences = [item[1] for item in data.original_seq]
    ptr = 0
    data_size = len(sequences)
    while ptr < data_size:
        if ptr+batch_size<data_size:
            batch_end = ptr+batch_size
        else:
            batch_end = data_size
        seq = np.zeros((batch_end-ptr, max_len),dtype=int)
        pos = np.zeros((batch_end-ptr, max_len),dtype=int)
        seq_len = []
        for n in range(0, batch_end-ptr):
            start = len(sequences[ptr + n]) > max_len and -max_len or 0
            end =  len(sequences[ptr + n]) > max_len and max_len or len(sequences[ptr + n])
            seq[n, :end] = sequences[ptr + n][start:]
            seq_len.append(end)
            pos[n, :end] = list(range(1,end+1))
        ptr=batch_end
        yield seq, pos, np.array(seq_len,int)