import numpy as np
from scipy.io import loadmat # for loading MATLAB files
import pandas as pd


def costfunction(Theta1, Theta2, X_with_bias, y_one_hot):

    # Feedforward
    a1 = X_with_bias # i.e. (5000, 401)

    z2 = Theta1.dot(a1.T)  # 25x401 * 401x5000 = 25x5000
    a2 = np.c_[np.ones((X_with_bias.shape[0], 1)), sigmoid(z2.T)]  # 5000x26

    z3 = Theta2.dot(a2.T)  # 10x26 * 26x5000 = 10x5000
    a3 = sigmoid(z3)  # 10x5000

    # Compute the cost
    J = -1 * (1 / X_with_bias.shape[0]) * np.sum((np.log(a3.T) * (y_one_hot) + np.log(1 - a3).T * (1 - y_one_hot)))

    return J


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


'''
Step 0: Do the following steps:
1. load the training dataset from 'data/digits.mat' and store as (X, y)
2. load the provided weights from 'data/weights.mat' and store as Theta1, Theta2
'''
dataset = loadmat('data/digits.mat')
print(dataset.keys())

weights = loadmat('data/weights.mat')
print(weights.keys())

X, y = dataset['X'], dataset['y']
print('X: {}'.format(X.shape))
print('y: {}'.format(y.shape))

Theta1, Theta2 = weights['Theta1'], weights['Theta2']
print('Theta1: {}'.format(Theta1.shape))
print('Theta2: {}'.format(Theta2.shape))


'''
Step 1: Augment the original matrix X with 1 (the intercept term)
'''

# Add constant for intercept
X_new = np.c_[np.ones((dataset['X'].shape[0], 1)), dataset['X']]
print('X_new (with intercept): {}'.format(X_new.shape), '\n')


'''
Step 2: Convert the original labels to one-hot vectors 
'''
y_one_hot = pd.get_dummies(y.ravel()).values


'''
Step 3: Compute the cost function
'''
print('The cost (without regularization) is', costfunction(Theta1, Theta2, X_new, y_one_hot))