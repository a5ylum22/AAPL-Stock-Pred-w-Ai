import pandas as pd
import numpy as np
import sklearn
from sklearn import preprocessing
import matplotlib.pyplot as plt

df = pd.read_csv('AAPL.csv')
df = df.drop(['Date', 'Adj Close', 'Volume'], axis = 1)

#df.head()
#df.describe().T

def feature_normalize(X, mean = np.zeros(1), std = np.zeros(1)):
    X = np.array(X)
    if len(mean.shape) == 1 or len(std.shape) == 1:
        mean = np.mean(X, axis = 0)
        std = np.std(X, axis = 0, ddof = 1)

    X = (X - mean)/std
    return X, mean, std

X_norm, mu, sigma = feature_normalize(df[['Open', 'High', 'Low']])

def compute_cost(X, y, theta):
    m = y.shape[0]
    h = X.dot(theta)
    J = (1/(2*m)) * ((h-y).T.dot(h-y))
    return J

def gradient_descent(X, y, theta, alpha, num_iters):
    m = y.shape[0]
    J_history = np.zeros(shape=(num_iters, 1))

    for i in range (0, num_iters):
        h = X.dot(theta)
        diff_hy = h-y

        delta = (1/m) * (diff_hy.T.dot(X))
        theta = theta - (alpha * delta.T)
        J_history[i] = compute_cost(X, y, theta)

    return theta, J_history

m = df.shape[0]
X = np.hstack((np.ones((m,1)), X_norm))
y = np.array(df.Close.values).reshape(-1,1)
theta = np.zeros(shape = (X.shape[1], 1))

#print(X)

alpha = [0.3, 0.1, 0.03, 0.01]
colors = ['b', 'r', 'g', 'c']
num_iters = 200

for i in range(0, len(alpha)):
    theta = np.zeros(shape = (X.shape[1], 1))
    theta, J_history = gradient_descent(X, y, theta, alpha[i], num_iters)
    plt.plot(range(len(J_history)), J_history, colors[i], label = 'alpha {}'.format(alpha[i]))
plt.xlabel('number of iterations');
plt.ylabel('cost J');
plt.title('selecting learning rates');
plt.legend()
plt.show()

iterations = 1500
alpha = 0.01
theta, loss = gradient_descent(X, y, theta, alpha, iterations)

print('theta found by gradient descent:')
print(theta)

Open = (14.055 - mu[0])/sigma[0]
High = (14.83 - mu[1])/sigma[1]
Low = (14.017 - mu[2])/sigma[2]
y_pred = theta[0] + theta[1] * Open + theta[2] * High + theta[3] * Low

print(y_pred)