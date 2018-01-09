import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage
from lr_utils import load_dataset
from sklearn.metrics import classification_report


def plot_test(x):
    fig, ax = plt.subplots(5, 10, sharex=True, sharey=True)
    for i in range(5):
        for j in range(10):
            ax[i, j].imshow(x[i*5+j])
    plt.show()


def sigmoid(z):
    return 1 / (1+np.exp(-z))



def cost_function(w, b, X, y):
    y_hat = sigmoid(w.T @ X + b)
    J = -(y * np.log(y_hat) + (1-y)*np.log(1-y_hat))
    return np.sum(J)


def gradient_decent(w, b, X, y, alpha, iters):
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


def predict(w, b, X, y):
    # for i in range(X.shape[1]):
    p = sigmoid(w.T@X + b)
    cat = p > 0.5
    cat = cat.astype('int')
    target_name = ['class0', 'class1']
    # report = classification_report(y.reshape(-1, 1), cat.reshape(-1, 1), target_names=target_name)
    # print(report)
    # correct_num = [1 if(a==1 and b==1) else 0 for (a, b) in zip(cat, y)]
    correct_num = [1 if((a == 1 and q == 1) or (a == 0 and q == 0)) else 0 for (a, q) in zip(cat.ravel(), y.ravel())]
    accrucy = np.sum(np.array(correct_num)) / len(correct_num)
    print(accrucy)

    # accrucy = correct_num / X.shape[1]
    #     if a < 0.5:
    #         print("It's not a cat")
    # plt.imshow(X.reshape(64, 64, 3))
    # plt.show()


train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()
train_set_x_orig = train_set_x_orig.reshape((train_set_x_orig.shape[0], -1)).T
test_set_x_orig = test_set_x_orig.reshape((test_set_x_orig.shape[0], -1)).T
train_set_x = train_set_x_orig / 255.
test_set_x = test_set_x_orig / 255.
b = 0
w = np.zeros((train_set_x.shape[0], 1))
cost = cost_function(w, b, train_set_x, train_set_y)
print(cost)
w, b, J = gradient_decent(w, b, train_set_x, train_set_y, 0.004, 500)
print(test_set_x.shape)
print(test_set_y.shape)
print(test_set_y[:, 5])
predict(w, b, test_set_x, test_set_y)

# plt.plot(J)
# plt.show()




#
# print(train_set_x_orig.shape)
# print(test_set_x_orig.shape)
# print(test_set_y.shape)
# print(classes)
# print(train_set_x_orig.shape)
# print(test_set_x_orig.shape)
# print(train_set_x_orig[0:5, 0])
# print(train_set_x[0:5, 0])

