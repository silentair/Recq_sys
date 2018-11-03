'''Singular Value Decomposition plus plus'''
import tensorflow as tf
import numpy as np

# 计算评分矩阵
def SVD_plusplus_pred(ui_mat,avg,n_i,n_u,n_k,lr = 0.01,lamb_u = 0.02,lamb_v = 0.02):

    U = tf.Variable(tf.random_normal(shape = [n_u, n_k]))
    V = tf.Variable(tf.random_normal(shape = [n_i, n_k]))
    U_bias = tf.Variable(tf.random_normal(shape = [n_u]))
    V_bias = tf.Variable(tf.random_normal(shape = [n_i]))
#svd++
    Y = tf.Variable(tf.random_normal(shape = [n_i, n_k]))

    u_idx = tf.placeholder(tf.int32,shape = [None])
    i_idx = tf.placeholder(tf.int32,shape = [None])
#svd++
    L = tf.placeholder(tf.float32,shape = [None])

    U_embed = tf.nn.embedding_lookup(U, u_idx)
    V_embed = tf.nn.embedding_lookup(V, i_idx)
    U_bias_embed = tf.nn.embedding_lookup(U_bias, u_idx)
    V_bias_embed = tf.nn.embedding_lookup(V_bias, i_idx)
#svd++
    Y_embed = tf.nn.embedding_lookup(Y, i_idx)

    rating_pred = tf.reduce_sum(Y_embed,axis = 0) / tf.sqrt(L) + U_embed
    rating_pred = tf.matmul(rating_pred, tf.transpose(V_embed))
    rating_pred = tf.add(rating_pred,U_bias_embed)
    rating_pred = tf.add(rating_pred,V_bias_embed)
    rating_pred = tf.add(rating_pred,avg)

    rating_real = tf.placeholder(tf.float32,shape = [None])

    loss_rat = tf.nn.l2_loss(rating_real - rating_pred)
    loss_reg_u = tf.multiply(lamb_u,tf.nn.l2_loss(U)) + tf.multiply(lamb_u,tf.nn.l2_loss(U_bias))
#svd++
    loss_reg_v = tf.multiply(lamb_v,tf.nn.l2_loss(V)) + tf.multiply(lamb_v,tf.nn.l2_loss(V_bias)) + tf.multiply(lamb_v,tf.nn.l2_loss(Y))

    loss = loss_rat + loss_reg_u + loss_reg_v

    optimizer_U = tf.train.AdamOptimizer(lr).minimize(loss,var_list=[U,U_bias])
    optimizer_V = tf.train.AdamOptimizer(lr).minimize(loss,var_list=[V,V_bias,Y])

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for step in range(5):
            for u in range(n_u):

                sess.run(optimizer_U, feed_dict={ u_idx:[u], i_idx:rat_idx[u], rating_real: [ui_mat[u][i] for i in rat_idx[u]], L:[len(rat_idx[u])]  })
                sess.run(optimizer_V, feed_dict={ u_idx:[u], i_idx:rat_idx[u], rating_real: [ui_mat[u][i] for i in rat_idx[u]], L:[len(rat_idx[u])]  })

                loss_ = sess.run(loss, feed_dict={ u_idx:[u], i_idx:rat_idx[u], rating_real: [ui_mat[u][i] for i in rat_idx[u]], L:[len(rat_idx[u])] })

                print('loss:',loss_)

        res_U = sess.run(U)
        res_V = sess.run(V)
        res_U_bias = sess.run(U_bias)
        res_V_bias = sess.run(V_bias)
        res_Y = sess.run(Y)

    y = [np.sum([res_Y[j] for j in u],axis = 0) for u in rat_idx]
    ra = [[np.dot(res_U[a]+y[a], res_V[b])+res_U_bias[a]+res_V_bias[b]+avg for b in range(n_i)] for a in range(n_u)]

    return ra