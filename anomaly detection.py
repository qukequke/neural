import scipy.io as sio
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from scipy.stats import multivariate_normal
import numpy as np
from sklearn.metrics import classification_report


def split_data(mat):
    """
    把原来的val数据一般分给test
    :return:
    """
    xval, xtest, yval, ytest = train_test_split(mat.get('Xval'), mat.get('yval').ravel(), test_size=0.5)
    return xval, xtest, yval, ytest


def load_data():
    mat = sio.loadmat('./data/ex8data1.mat')
    print(mat.keys())
    return mat


def plot_multi_norml(data):
    """
    画原始的等高线图
    """
    data_cov = np.cov(data.T)
    gaosi = multivariate_normal(np.mean(data, axis=0), data_cov)
    x, y = np.meshgrid(np.arange(0, 30, .01), np.arange(0, 30, .01))
    zuobiao = np.dstack((x, y)) #转变为正常的（x,y)坐标传进去
    z = gaosi.pdf(zuobiao)
    plt.contourf(x, y, z, cmap='Blues')
    plt.scatter(data[:, 0], data[:, 1], c='r', s=5)
    plt.show()


def select_best_threshold(data, xval, yval):
    """
    选择 阈值
    一个一个试，找到在cv集中f1score最高的阈值 ，
    return：阈值 和最高的f1score
    """
    data_cov = np.cov(data.T)
    gaosi = multivariate_normal(np.mean(data, axis=0), data_cov)
    prob = gaosi.pdf(xval)
    epsilons = np.linspace(np.min(prob), np.max(prob), 10000)
    print(epsilons)
    fs = []
    for threshold in epsilons:
        y_pred = (prob <= threshold).astype('int')
        a = f1_score(yval, y_pred)
        fs.append(a)
        index = np.argmax(fs)
    return epsilons[index], fs[index]


def plot_best_multi_norml(data, parameter, xtest, ytest):
    """
    画出异常点 和 测试集
    :param data:
    :param parameter:
    :param xtest:
    :param ytest:
    :return:
    """
    data_cov = np.cov(data.T)
    gaosi = multivariate_normal(np.mean(data, axis=0), data_cov)
    x1_min, x1_max = np.min(xtest[:, 0]), np.max(xtest[:, 0])
    x2_min, x2_max = np.min(xtest[:, 1]), np.max(xtest[:, 1])
    print(x1_min, x1_max, x2_min, x2_max)
    x, y = np.meshgrid(np.arange(x1_min, x1_max, .01), np.arange(x2_min, x2_max, .01))
    zuobiao = np.dstack((x, y)) #转变为正常的（x,y)坐标传进去
    z = gaosi.pdf(zuobiao)

    y_prob = gaosi.pdf(xtest)
    y_pred_bool = (y_prob < parameter)
    y_pred = y_pred_bool.astype('int')
    report = classification_report(ytest, y_pred)
    print(report)
    print(xtest[y_pred_bool])
    plt.contourf(x, y, z, cmap='Blues')
    plt.scatter(xtest[:, 0], xtest[:, 1], c='r', s=5)
    plt.scatter(xtest[y_pred_bool, 0], xtest[y_pred_bool, 1], marker='x', s=40)
    plt.show()

mat = load_data()
data = mat.get("X")
print(data.shape)
xval, xtest, yval, ytest = split_data(mat)
# plot_multi_norml(xtest)
parameter, value = select_best_threshold(data, xval, yval)

data = np.r_[data, xval]
plot_best_multi_norml(data, parameter, xtest, ytest)


