''' Here are some evaluations '''
import numpy as np

''' similarity '''
#欧氏距离
def Euclidean_sim(vector_a,vector_b):
    return 1 / (1 + np.linalg.norm(vector_a-vector_b))

#皮尔逊相关系数
def Pearson_sim(vector_a,vector_b):
    #var_a = np.std(tf_a)
    #var_b = np.std(tf_b)
    #cov_ab = np.cov(a,b,ddof=0)

    return 0.5 * np.corrcoef(vector_a,vector_b)[0][1] + 0.5

#余弦距离
def Cosin_sim(vector_a,vector_b):
    return np.sum(vector_a * vector_b) / (np.linalg.norm(vector_a) * np.linalg.norm(vector_b))