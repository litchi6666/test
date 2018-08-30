import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris


def loaddata():
    iris = load_iris()
    x = iris['data']
    y = iris['target']

    y_ =[]
    for cls in y:
        temp = [0,0,0]
        temp[cls] = 1
        y_ .append(temp)
    y_ = np.array(y_)
    return x, y_


class MulitLR():
    def __init__(self):
        self.learning_rate = 0.01
        self.iteration = 200

    def softmax(self,x,w):
        z = np.dot(x,w.transpose())
        z = np.exp(z)

        sum_z = np.sum(z,axis=1)
        for i in range(len(sum_z)):
            z[i] = z[i] / sum_z[i]

        return z  # 一行代表一个样本属于每一个类别的概率值，取argmax可以得到预测值

    def grad_descent(self, x, y, w):

        y_p = self.softmax(x, w)
        grad = np.dot(x.transpose(), y - y_p).transpose()
        w = w + self.learning_rate * grad

        return w

    def log_liklihood(self, x, y, w):
        softmax = np.log(self.softmax(x, w))
        loss = - np.sum(softmax * y)

        return loss

    def train(self):
        x, y = loaddata()
        n = len(x[0])   # 样本的特征维度编号
        k = len(y[0])   # 类别的个数编号
        w = np.zeros(shape=(k,n))

        for i in range(self.iteration):
            loss = self.log_liklihood(x, y, w)
            w = self.grad_descent(x, y, w)
            acc = self.acc(x, y, w)
            self.log_liklihood(x, y, w)
            print('迭代次数:%s,loss:%s,acc:%s' % (i, loss, acc))

    def acc(self,x,y,w):
        y = np.argmax(y,1)
        y_hat = np.argmax(self.softmax(x,w),1)

        acc = 0.0
        for i in range(len(y)):

            if y[i] == y_hat[i]:
                acc += 1
        return acc/len(y)


if __name__ == '__main__':
    lr = MulitLR()
    lr.train()



