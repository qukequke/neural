import numpy as np
import scipy.io as sio
import scipy.optimize as opt
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_data():
    """for ex5
    d['X'] shape = (12, 1)
    pandas has trouble taking this 2d ndarray to construct a dataframe, so I ravel
    the results
    """
    d = sio.loadmat('ex5data1.mat')
    return map(np.ravel, [d['X'], d['y'], d['Xval'], d['yval'], d['Xtest'], d['ytest']])


def cost(theta, X, y):
    # print(X.shape)
    m = X.shape[0]
    cost = (1/(2*m)) * sum(np.square(X @ theta - y))
    return cost


def regulaztion_cost(theta, X, y, lamda=1):
    m = X.shape[0]
    total_cost = cost(theta, X, y) + (lamda/(2*m)) * sum(np.square(theta[1:, ]))
    return total_cost


def gradient(theta, X, y):
    m = X.shape[0]
    return (1/m) * (X.T @ (X @ theta - y))


def regulazition_gtadient(theta, X, y, lamda=1):
    m = X.shape[0]
    regu = theta.copy()
    regu[0] = 0
    return gradient(theta, X, y) + (lamda/m) * regu


def linear_regression(theta, X, y, I=1):
    res = opt.minimize(fun=regulaztion_cost, x0=theta, args=(X, y, I), method='TNC', jac=regulazition_gtadient)
    return res


def plot_m_change(theta, X, y, I=0):
    train_cost, cv_cost = [], []
    m = X.shape[0]
    for i in range(1, m+1):
        res = linear_regression(theta, X[:i, :], y[:i], I)
        # tc = regulaztion_cost(res.x, X[:i, :], y[:i])
        # cv = regulaztion_cost(res.x, Xval, yval)
        tc = cost(res.x, X[:i, :], y[:i])
        cv = cost(res.x, Xval, yval)
        train_cost.append(tc)
        cv_cost.append(cv)
    print(cv_cost)
    plt.plot(np.arange(1, m+1), train_cost, label='traning cost')
    plt.plot(np.arange(1, m+1), cv_cost, label='cv cost')
    plt.legend(loc=1)
    plt.show()


def normolize_feature(df):
    return df.apply(lambda column: (column - column.mean()) / column.std())


def prepare_poly_feature(*args, power):
    def inner(x):
        df = poly_feature(x, power)
        df_array = normolize_feature(df).as_matrix()
        return np.insert(df_array, 0, np.ones(df_array.shape[0]), axis=1)
    return [inner(x) for x in args]


def poly_feature(X, power, as_array=False):
    data = {'f{}'.format(i): np.power(X, i) for i in range(1, power+1)}
    df = pd.DataFrame(data)
    return df.as_matrix() if as_array else df


def plot_lamda_change(theta, X, y):
    cv_cost, tr_cost = [], []
    I_candidata = [0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10]
    for i in I_candidata:
        res = linear_regression(theta, X, y, i)
        tr = cost(res.x, X, y)
        cv = cost(res.x, Xval, yval)
        cv_cost.append(cv)
        tr_cost.append(tr)
    print(cv_cost)
    plt.plot(I_candidata, tr_cost, label='tr')
    plt.plot(I_candidata, cv_cost, label='cv')
    plt.legend(loc=2)
    plt.show()

X, y, Xval, yval, Xtest, ytest = load_data()
print(X.shape)
print(y.shape)
df = pd.DataFrame({'water_level': X, 'flow': y})
aa, Xval, Xtest = prepare_poly_feature(X, Xval, Xtest, power=8)
print(aa[:3, :])
# X, Xval, Xtest = [np.insert(x.reshape(x.shape[0], 1), 0, np.ones(x.shape[0]), axis=1) for x in (X, Xval, Xtest)]
# aa = aa[0]
theta = np.ones(aa.shape[1])
# res = linear_regressionG(theta, aa, y)
plot_m_change(theta, aa, y, 0)
plot_lamda_change(theta, aa, y)
I_candidata = [0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10]
theta1 = np.ones(1)
for l in I_candidata:
    theta = linear_regression(theta, aa, y, l).x
    print('test cost(l={}) = {}'.format(l, cost(theta, Xtest, ytest)))

