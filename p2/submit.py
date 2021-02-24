import numpy as np
# DO NOT add any library

def train_transformation(X, Y,R, train_steps=100, learning_rate=0.0003):
    #please write your code here
    # find best R using Gradient Descent
    #WRITE your code here
    n,m = X.shape
    for i in range(train_steps):
        y_hat = np.matmul(X,R)
        diff = Y-y_hat
        loss = (1/n)*(np.sum(diff**2))
        grad = (2/n) * np.matmul(np.transpose(X),(np.matmul(X,R)-Y))
        R -= learning_rate*grad
    #END of your code
    return R
    
def nearest_neighbor(v, candidates, k=1):
    # find k best similar vectors index. please sort them in order max to min and return index
    # for your similarity function please use cosine similarity
    similarity_l = []
    #WRITE your code here
    normv = np.linalg.norm(v)
    similarity_l = [np.dot(v,c)/(normv*(np.linalg.norm(c))) for c in candidates]
    #END of your code
    sorted_ids = np.argsort(similarity_l)

    k_idx = sorted_ids[-k:]
    return k_idx

