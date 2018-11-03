'''Singular Value Decomposition'''
import tensorflow as tf
import numpy as np
import Metric.similarity as sm

# 降维的维度
def k_Sigma(sigma,percentage = 0.9):
    sigma_pow = sigma**2
    sigma_sum = np.sum(sigma_pow)
    
    sigma_sum_k = 0
    k = 0
    for i in sigma:
        sigma_sum_k = sigma_sum_k + i**2
        k = k + 1
        if sigma_sum_k >= sigma_sum * percentage:
            break

    return k

# u_idx对i_list中的item的评分
def SVD_pred_with_sim(ui_mat,u_idx,i_list,sim_metric = 'Cosin'):
    if sim_metric not in ['Cosin','Euclidean','Pearson']:
        print('Wrong parameter of sim_metric')
        exit(1)

    # svd
    u,s,v = np.linalg.svd(ui_mat)
    # numbers of items and users
    n_i = len(v)
    n_u = len(u)
    # dimension to reduce
    k = k_Sigma(s)
    # matrix u and sigma
    s_k = np.mat(np.diag(s[:k]))
    u_k = np.mat(u[:,:k])
    # matrix v
    tt = np.matmul(np.mat(ui_mat).T,u_k)
    v_k = np.matmul(tt,s_k.I)
    # calculate ratings of items in i_list
    print('estimate for user'+str(u_idx))
    ra=[]
    for i_idx in i_list:
        sim_sum = 0
        rating_sum = 0
        for ii_idx in range(n_i):
            if ii_idx == i_idx or ui_mat[u_idx,ii_idx] == 0:
                continue
            
            if sim_metric == 'Cosin':
                sim = sm.Cosin_sim(np.array(v_k[ii_idx,:]),np.array(v_k[i_idx,:]))
            if sim_metric == 'Euclidean':
                sim = sm.Euclidean_sim(np.array(v_k[ii_idx,:]),np.array(v_k[i_idx,:]))
            if sim_metric == 'Pearson':
                sim = sm.Pearson_sim(np.array(v_k[ii_idx,:]),np.array(v_k[i_idx,:]))

            sim_sum = sim_sum + sim
            rating_sum = rating_sum + sim * ui_mat[u_idx,ii_idx]

        if sim_sum == 0:
            ra.append(0)
        else:
            ra.append(rating_sum / sim_sum)

    return ra

# 计算评分矩阵
def SVD_pred(ui_mat,avg,n_i,n_u,n_k,lr = 0.01,lamb_u = 0.02,lamb_v = 0.02):
    U = tf.Variable(tf.random_normal(shape = [n_u, n_k]))
    V = tf.Variable(tf.random_normal(shape = [n_i, n_k]))
    U_bias = tf.Variable(tf.random_normal(shape = [n_u]))
    V_bias = tf.Variable(tf.random_normal(shape = [n_i]))

    u_idx = tf.placeholder(tf.int32,shape = [None])
    i_idx = tf.placeholder(tf.int32,shape = [None])

    U_embed = tf.nn.embedding_lookup(U, u_idx)
    V_embed = tf.nn.embedding_lookup(V, i_idx)
    U_bias_embed = tf.nn.embedding_lookup(U_bias, u_idx)
    V_bias_embed = tf.nn.embedding_lookup(V_bias, i_idx)

    rating_pred = tf.matmul(U_embed, tf.transpose(V_embed))
    rating_pred = tf.add(rating_pred,U_bias_embed)
    rating_pred = tf.add(rating_pred,V_bias_embed)
    rating_pred = tf.add(rating_pred,avg)

    rating_real = tf.placeholder(tf.float32,shape = [None])

    loss_rat = tf.nn.l2_loss(rating_real - rating_pred)
    loss_reg_u = tf.multiply(lamb_u,tf.nn.l2_loss(U)) + tf.multiply(lamb_u,tf.nn.l2_loss(U_bias))
    loss_reg_v = tf.multiply(lamb_v,tf.nn.l2_loss(V)) + tf.multiply(lamb_v,tf.nn.l2_loss(V_bias))

    loss = loss_rat + loss_reg_u + loss_reg_v

    optimizer_U = tf.train.AdamOptimizer(lr).minimize(loss,var_list=[U,U_bias])
    optimizer_V = tf.train.AdamOptimizer(lr).minimize(loss,var_list=[V,V_bias])

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for step in range(5):
            for u in range(n_u):

                sess.run(optimizer_U, feed_dict={ u_idx:[u], i_idx:rat_idx[u], rating_real: [ui_mat[u][i] for i in rat_idx[u]]  })
                sess.run(optimizer_V, feed_dict={ u_idx:[u], i_idx:rat_idx[u], rating_real: [ui_mat[u][i] for i in rat_idx[u]]  })

                loss_ = sess.run(loss, feed_dict={ u_idx:[u], i_idx:rat_idx[u], rating_real: [ui_mat[u][i] for i in rat_idx[u]] })

                print('loss:',loss_)

        res_U = sess.run(U)
        res_V = sess.run(V)
        res_U_bias = sess.run(U_bias)
        res_V_bias = sess.run(V_bias)

        ra = [[np.dot(res_U[a], res_V[b])+res_U_bias[a]+res_V_bias[b]+avg for b in range(n_i)] for a in range(n_u)]

        return ra