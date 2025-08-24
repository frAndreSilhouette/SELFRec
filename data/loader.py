import os.path
from os import remove
from re import split


class FileIO(object):
    def __init__(self):
        pass

    @staticmethod
    def write_file(dir, file, content, op='w'):
        if not os.path.exists(dir):
            os.makedirs(dir)
        with open(dir + file, op) as f:
            f.writelines(content)

    @staticmethod
    def delete_file(file_path):
        if os.path.exists(file_path):
            remove(file_path)

    @staticmethod
    def load_data_set(file, rec_type):
        if rec_type == 'graph':
            data = []
            with open(file) as f:
                for line in f:
                    # # ==========================
                    # # 原始版本
                    # items = split(' ', line.strip())
                    # user_id = items[0]
                    # item_id = items[1]
                    # weight = items[2]
                    # data.append([user_id, item_id, float(weight)])
                    # # ===========================

                    # ===========================
                    # 新版本：加入时间信息
                    # 原始版本
                    items = split(' ', line.strip())
                    user_id = items[0]
                    item_id = items[1]
                    itt = int(items[2]) # itt是距离上次购买的天数(int)
                    scale = float(items[3])
                    shape = float(items[4])
                    weight = float(items[5])
                    current_itt = int(items[6])
                    data.append([user_id, item_id, itt, scale, shape, weight, current_itt])
                    # ===========================

        if rec_type == 'sequential':
            data = {}
            with open(file) as f:
                for line in f:
                    items = split(':', line.strip())
                    seq_id = items[0]
                    data[seq_id]=items[1].split()
        return data

    @staticmethod
    def load_user_list(file):
        user_list = []
        print('loading user List...')
        with open(file) as f:
            for line in f:
                user_list.append(line.strip().split()[0])
        return user_list

    @staticmethod
    def load_social_data(file):
        social_data = []
        print('loading social data...')
        with open(file) as f:
            for line in f:
                items = split(' ', line.strip())
                user1 = items[0]
                user2 = items[1]
                if len(items) < 3:
                    weight = 1
                else:
                    weight = float(items[2])
                social_data.append([user1, user2, weight])
        return social_data
