'''embedding'''
import numpy as np
import Algorithm.rating.SVD as svd

# item and user to vector by svd
def SVD_ItemUser2vec(ui_mat,factor_num=None):
    # svd
    u,s,v = np.linalg.svd(ui_mat)

    # dimension to reduce
    if factor_num is None:
        k = svd.k_Sigma(s)
    else:
        k = factor_num

    # matrix u and sigma
    sigma = np.mat(np.diag(s[:k]))
    user_vec = np.mat(u[:,:k])
    # matrix v
    tt = np.matmul(np.mat(ui_mat).T,user_vec)
    item_vec = np.matmul(tt,sigma.I)

    return user_vec.tolist(),item_vec.tolist()