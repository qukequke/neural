import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage
from lr_utils import load_dataset
from sklearn.metrics import classification_report


def plot_test(x):
    """
    可以看一下图片：这里写的默认5*10的
    :param x: 4维  每一行都是一个图片
    """
    fig, ax = plt.subplots(5, 10, sharex=True, sharey=True)
    for i in range(5):
        for j in range(10):
            ax[i, j].imshow(x[i*5+j])
    plt.show()


def sigmoid(z):
    return 1 / (1+np.exp(-z))


def cost_function(w, b, X, y):
    m = X.shape[1]
    y_hat = sigmoid(w.T @ X + b)
    J = -(y * np.log(y_hat) + (1-y)*np.log(1-y_hat)) / m
    return np.sum(J)


def gradient_decent(w, b, X, y, alpha, iters):
    b = 0
    w = np.zeros((train_set_x.shape[0], 1))
    m = X.shape[1]
    J = np.zeros(iters)
    for i in range(iters):
        J[i] = cost_function(w, b, X, y)
        A = sigmoid(w.T@X+b)
        dz = (A-y)
        dw = (X @ dz.T) / m
        db = (np.sum(dz)) / m
        w -= alpha * dw
        b -= alpha * db
    return w, b, J


def predict(w, b, X, y=None):
    """
    做预测 有两种方式 一种是调用sklearn.metrics classfication
    还有一种是 手写的
    """
    p = sigmoid(w.T@X + b)
    if p.shape[1] > 1:
        cat = p > 0.5
        cat = cat.astype('int')
        # target_name = ['class0', 'class1']
        # report = classification_report(y.reshape(-1, 1), cat.reshape(-1, 1), target_names=target_name)
        # print(report)
        correct_num = [1 if((a == 1 and q == 1) or (a == 0 and q == 0)) else 0 for (a, q) in zip(cat.ravel(), y.ravel())]
        accrucy = (np.sum(np.array(correct_num)) / len(correct_num)) * 100
        if y.shape[1] == 209:
            print('train accrucy=' + str(accrucy) + '%')
        else:
            print('test accrucy=' + str(accrucy) + '%')
    else:
        if p > 0.5:
            print("It's a cat")
        if p < 0.5:
            print("It' not a cat")
        plt.imshow(X.reshape(64, 64, 3))
        plt.show()


def try_lamda(w, b, train_x, train_y, test_x, test_y):
    lamda = [0.01, 0.001, 0.003, 0.0001]
    J = np.zeros((len(lamda), 500))
    for j, i in enumerate(lamda):
        w1, b1, J[j] = gradient_decent(w, b, train_x, train_y, i, 500)
        print('')
        print('alpha=' + str(i))
        predict(w1, b1, train_x, train_y )
        predict(w1, b1, test_x, test_y)

    plt.plot(J[0, ], c='r', label='0.01')
    plt.plot(J[1, ], c='b', label='0.001')
    plt.plot(J[2, ], label='0.003')
    plt.plot(J[3, ], label='0.0001')
    plt.legend(loc=1)
    plt.xlabel('iterations')
    plt.ylabel('cost')
    plt.show()


train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()
train_set_x_orig = train_set_x_orig.reshape((train_set_x_orig.shape[0], -1)).T
test_set_x_orig = test_set_x_orig.reshape((test_set_x_orig.shape[0], -1)).T
train_set_x = train_set_x_orig / 255.
test_set_x = test_set_x_orig / 255.

b = 0
w = np.zeros((train_set_x.shape[0], 1))
try_lamda(w, b, train_set_x, train_set_y, test_set_x, test_set_y)

#测试自己的图片
# w, b, J = gradient_decent(w, b, train_set_x, train_set_y, 0.01, 500)
# for i in range(8, 11):
#     folder_name = str(i) + '.jpg'
#     my_picture = Image.open('./my_search/' + folder_name)
#     my_picture = np.array(my_picture)
#     predict(w, b, my_picture.reshape(-1, 1))


