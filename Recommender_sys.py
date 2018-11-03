import numpy as np
import tensorflow as tf
import Algorithm.rating.MF as mf
import Datainfo.acquisition as ac

'''parameters
ui_mat = ui_matrix
avg = rating_mean
n_i = i_num
n_u = u_num
n_k = k
lr = 0.01
lamb_u = 0.02
lamb_v = 0.02
'''

#ra = SVD_plusplus_pred(ui_matrix,rating_mean,i_num,u_num,k)
#print(ra[2][13:30])

data = ac.get_FilmTrust()

ui_matrix,is_rating = ac.get_User_Item_matrix(data)

#ra = mf.MF_pred(ui_matrix,is_rating)
#print(ra[2][13:30])

#ilist = [13,14,15,16,1,18,19,20]
#ra_2_to_ilist = svd.SVD_pred_with_sim(ui_matrix,2,ilist)
#print(ra_2_to_ilist)

#import Algorithm.embedding as eb

#u,v = eb.SVD_ItemUser2vec(ui_matrix)
