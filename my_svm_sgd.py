import numpy as np
import matplotlib.pyplot as plt


def gen_data():
    x_train = []
    y_train = []

    x_test = []
    y_test = []

    for i in range(150):
        x_1 = np.random.randint(0, 30)
        x_2 = np.random.randint(0, 30)

        x = [x_1, x_2]
        y = 1

        if i < 100:
            x_train.append(x)
            y_train.append(y)

        else:
            x_test.append(x)
            y_test.append(y)

    for i in range(150):
        x_1 = np.random.randint(20, 50)
        x_2 = np.random.randint(20, 50)

        x = [x_1, x_2]
        y = -1

        if i < 100:
            x_train.append(x)
            y_train.append(y)

        else:
            x_test.append(x)
            y_test.append(y)

    # for i in range(len(y_train)):
    #     if y_train[i] == 1:
    #         plt.plot(x_train[i][0], x_train[i][1], 'ob')
    #     else:
    #         plt.plot(x_train[i][0], x_train[i][1], 'or')
    x = np.array(x_train)
    y = np.array(y_train)
    x_ = np.array(x_test)
    y_ = np.array(y_test)

    return x, y, x_, y_


class svm_sgd(object):
    def __init__(self,c=4.0):
        self.x_train ,self.y_train, self.x_test, self.y_test = gen_data()
        self.C = c
        self.w = np.zeros(shape=[len(self.x_train[0])])
        self.b = 0.0
        self.rate = 0.001
        self.iter = 2000

    def loss(self):
        loss = 0.0
        for i in range(len(self.y_train)):
            y_i = self.y_train[i]
            x_i = self.x_train[i]
            temp = 1 - y_i * (np.dot(self.w, x_i.transpose()) + self.b)
            loss += max(0,temp)
        loss += self.C * np.dot(self.w,self.w.transpose())
        return loss

    def sgd(self,sample_idx):
        i = sample_idx
        y_i = self.y_train[i]
        x_i = self.x_train[i]
        temp = 1 - y_i * (np.dot(self.w, x_i.transpose()) + self.b)

        if temp > 0:
            grad_w = - y_i * x_i
            grad_b = - y_i
        else:
            grad_w = 0
            grad_b = 0

        self.w = self.w - self.rate * grad_w
        self.b = self.b - self.rate * grad_b

    def acc(self):
        right_nums = 0.0
        for i in range(len(self.x_train)):
            y_i = self.y_train[i]
            x_i = self.x_train[i]

            temp = np.dot(self.w, x_i) + self.b
            y_hat = 1 if temp >= 0 else -1

            if y_i == y_hat:
                right_nums += 1
        return right_nums / len(self.x_train)

    def train(self):
        it = 0
        while it < self.iter:
            it += 1
            for i in range(len(self.x_train)):
                self.sgd(i)

            loss = self.loss()
            acc = self.acc()
            print('iter:%s, loss:%s, acc:%s' % (it, loss, acc))
        self.show()

    def show(self):
        for i in range(len(self.x_train)):
            if self.y_train[i] == 1:
                plt.plot(self.x_train[i][0],self.x_train[i][1],'ob')
            else:
                plt.plot(self.x_train[i][0], self.x_train[i][1], 'or')

        for i in range(50):
            y = (-self.w[0] * i - self.b) / self.w[1]
            plt.plot(i, y, '*g')
        plt.show()


if __name__ =='__main__':
    svm = svm_sgd()
    svm.train()