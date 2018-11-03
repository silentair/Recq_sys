'''Matrix Factorization'''
import tensorflow as tf

# 计算评分矩阵
def MF_pred(ui_mat,is_rating,factor_num = 10,lr = 0.15):

    u_num = len(ui_mat)
    i_num = len(ui_mat[0])

    U = tf.Variable(tf.random_normal(shape=[u_num,factor_num],dtype=tf.float32))
    V = tf.Variable(tf.random_normal(shape=[factor_num,i_num],dtype=tf.float32))

    pred_rating = tf.matmul(U,V)
    #print(pred_rating)
    loss = tf.nn.l2_loss((ui_mat - pred_rating)*is_rating)

    optimzer = tf.train.AdamOptimizer(lr).minimize(loss)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(10):
            sess.run(optimzer)
            loss_ = sess.run(loss)
            print(loss_)
    
        ra = sess.run(pred_rating)

    return ra