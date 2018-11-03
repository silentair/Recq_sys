'''Acquiring data'''
import numpy as np

# 获取FilmTrust数据集（以[u,i,r]三元组的形式）
def get_FilmTrust():
    f = open('dataset/FilmTrust/ratings.txt')
    row_data = f.read().split()

    tup_len = len(row_data) // 3
    data = []
    for i in range(tup_len):
        data.append(row_data[i*3:i*3+3])

    return data

# 获取user,item的数目
def get_User_Item_num(data):
    user = []
    item = []
    for uir in data:
        u_idx = int(uir[0])
        i_idx = int(uir[1])
        if u_idx not in user:
            user.append(u_idx)
        if i_idx not in item:
            item.append(i_idx)
    u_num = max(user)
    i_num = max(item)

    return u_num,i_num

# 构建user-item评分矩阵
def get_User_Item_matrix(data):
    u_num,i_num = get_User_Item_num(data)
    ui_matrix = []
    ui_list = []
    for i in range(i_num):
        ui_list.append(0)
    for i in range(u_num):
        ui_matrix.append(ui_list[:])

    for uir in data:
        u_idx = int(uir[0]) - 1
        i_idx = int(uir[1]) - 1
        ui_matrix[u_idx][i_idx] = float(uir[2])

    # user-item中有评分的项目
    is_rating = np.array(np.array(ui_matrix) > 0,dtype=int).tolist()

    return ui_matrix,is_rating

# user所评分过得item
def get_User_rated_Item(is_rating):
    rat_idx = []
    for u in is_rating:
        temp = []
        for i in range(i_num):
            if u[i] == 1:
                temp.append(i)
        rat_idx.append(temp)

    return rat_idx

# item的平均分
def get_rating_mean(data):
    rating_sum = 0
    rating_num = 0
    for l in data:
        rating_num = rating_num + 1
        rating_sum = rating_sum + float(l[2])

    return rating_sum / rating_num