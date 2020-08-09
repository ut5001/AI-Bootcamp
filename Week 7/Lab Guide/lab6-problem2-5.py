import numpy as np
from scipy.io import loadmat # for loading MATLAB files
import pandas as pd


def costfunction(Theta1, Theta2, X_with_bias, y_one_hot, lambda_param):

    # Feedforward
    a1 = X_with_bias # 5000x401

    z2 = Theta1.dot(a1.T)  # 25x401 * 401x5000 = 25x5000
    a2 = np.c_[np.ones((X_with_bias.shape[0], 1)), sigmoid(z2.T)]  # 5000x26

    z3 = Theta2.dot(a2.T)  # 10x26 * 26x5000 = 10x5000
    a3 = sigmoid(z3)  # 10x5000

    # Compute the cost
    number_of_examples = X_with_bias.shape[0]

    J = -1 * (1 / number_of_examples) * np.sum((np.log(a3.T) * (y_one_hot) + np.log(1 - a3).T * (1 - y_one_hot))) + \
        (lambda_param/(2 * number_of_examples))*(np.sum(np.square(Theta1[:,1:])) + np.sum(np.square(Theta2[:,1:])))

    # Compute the gradients
    delta3 = a3.T - y_one_hot  # 5000x10
    delta2 = Theta2[:, 1:].T.dot(delta3.T) * sigmoidGradient(z2)  # 25x10 *10x5000 * 25x5000 = 25x5000

    big_delta1 = delta2.dot(a1)  # 25x5000 * 5000x401 = 25x401
    big_delta2 = delta3.T.dot(a2)  # 10x5000 *5000x26 = 10x26

    # DON'T regularize the fist column of Theta1 and Theta2
    Theta1_ = np.c_[np.ones((Theta1.shape[0], 1)), Theta1[:, 1:]]
    Theta2_ = np.c_[np.ones((Theta2.shape[0], 1)), Theta2[:, 1:]]

    theta1_grad = big_delta1 / number_of_examples + (Theta1_ * lambda_param) / number_of_examples
    theta2_grad = big_delta2 / number_of_examples + (Theta2_ * lambda_param) / number_of_examples

    return J, theta1_grad, theta2_grad


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def sigmoidGradient(z):
    return sigmoid(z) * (1 - sigmoid(z))


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
Step 3: Compute the cost function with lambda = 1
'''
print('The cost (without regularization) is', costfunction(Theta1, Theta2, X_new, y_one_hot, 1)[0], '\n')


'''
Step 4: Test the implemented sigmoidGradient. 
For large values (both positive and negative), the gradient should be close to 0. 
Furthermore, when z = 0, the gradient should be exactly 0.25.
'''
print('Compute the sigmoidGradient for -1, -0.5, 0, 0.5, 1 :', '\n', [sigmoidGradient(z) for z in [-1, -0.5, 0, 0.5, 1]])
