import numpy as np
import matplotlib.pyplot as plt


def generate_data():
    def show(x, y, w):
        for i in range(len(x)):
            if y[i] == 0:
                plt.plot(x[i][0], x[i][1], 'or')
            else:
                plt.plot(x[i][0], x[i][1], 'ob')
        for i in range(50):
            y = (-w[0] * i - w[2]) / w[1]
            plt.plot(i, y, '*g')
        plt.show()
    X_train = []
    Y_train = []
    X_test = []
    Y_test = []

    w = [2,3,-100]
    for i in range(130):
        x_1 = np.random.randint(0,50)
        x_2 = np.random.randint(0,50)

        X = [x_1, x_2]
        if w[0] * x_1 + w[1] * x_2 + w[2] > 0:
            y = 1
        else:
            y = 0

        if i < 100:
            X_train.append(X)
            Y_train.append(y)
        else:
            X_test.append(X)
            Y_test.append(y)

    # show(X_train,Y_train,w)
    # show(X_train, Y_train, [-0.25 ,-0.35 ,11.60])

    return X_train, Y_train, X_test, Y_test


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
        y = 0

        if i < 100:
            x_train.append(x)
            y_train.append(y)

        else:
            x_test.append(x)
            y_test.append(y)

    for i in range(len(y_train)):
        if y_train[i] == 1:
            plt.plot(x_train[i][0], x_train[i][1], 'ob')
        else:
            plt.plot(x_train[i][0], x_train[i][1], 'or')
    return x_train, y_train, x_test, y_test


class lr():
    def __init__(self):
        self.alpha = 0.01
        self.iteration = 10000
        self.x_train, self.y_train, self.x_test, self.y_test = gen_data()

    def sigmoid(self, x):
        return 1.0 - 1.0/(1 + np.exp(- x))

    def logliklihood(self, x, y, w):

        w_mult_x = np.dot(w,x)

        loss_1 = -(np.dot((1 - y), np.log(1 - self.sigmoid(w_mult_x))) + np.dot(y, np.log(self.sigmoid(w_mult_x))))
        loss_2 = np.log(1 + np.exp(w_mult_x)) - y*w_mult_x
        #loss_1 = np.dot(y,w_mult_x) - np.log(1 + np.exp(w_mult_x))
        return loss_1, loss_2

    def grad_descent(self, x, y, w):

        grad = np.dot((y - self.sigmoid(np.dot(w, x))), x)
        return w - self.alpha * grad

    def train(self):
        x_t = []
        for x in self.x_train:
            x.append(1)
            x_t.append(x)
        x = np.array(x_t)
        y = np.array(self.y_train)
        w = np.ones(x[0].shape).reshape(-1)

        print(x.shape, y.shape, w.shape)

        sample_numbers = len(x)
        for k in range(self.iteration):
            for i in range(sample_numbers):
                x_i = x[i]
                y_i = y[i]

                loss_1,loss_2 = self.logliklihood(x_i, y_i, w)

                w = self.grad_descent(x_i, y_i, w)
            # plt.plot(k,loss_1,'og')
            # plt.plot(k, loss_2, 'ob')
            print('第%s次迭代：loss：%s,%s,w：%s' % (k, loss_1,loss_2 , w))

        self.show(x, y, w)
        #plt.show()

    def show(self, x, y, w):
        for i in range(len(x)):
            if y[i] == 0:
                plt.plot(x[i][0], x[i][1], 'or')
            else:
                plt.plot(x[i][0], x[i][1], 'ob')
        for i in range(50):
            y = (-w[0] * i - w[2]) / w[1]
            plt.plot(i, y, '*g')
        plt.show()


if __name__ == '__main__':
    logr = lr()
    logr.train()
