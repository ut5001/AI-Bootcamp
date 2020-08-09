import numpy as np
from scipy.io import loadmat  # for loading MATLAB files
import pandas as pd


def costFunction(Theta1, Theta2, X_with_bias, y_one_hot, lambda_param):
    # Feedforward
    a1 = X_with_bias  # 5000x401

    z2 = Theta1.dot(a1.T)  # 25x401 * 401x5000 = 25x5000
    a2 = np.c_[np.ones((X_with_bias.shape[0], 1)), sigmoid(z2.T)]  # 5000x26

    z3 = Theta2.dot(a2.T)  # 10x26 * 26x5000 = 10x5000
    a3 = sigmoid(z3)  # 10x5000

    # Compute the cost
    number_of_examples = X_with_bias.shape[0]

    J = -1 * (1 / number_of_examples) * np.sum((np.log(a3.T) * (y_one_hot) + np.log(1 - a3).T * (1 - y_one_hot))) + \
        (lambda_param / (2 * number_of_examples)) * (
                    np.sum(np.square(Theta1[:, 1:])) + np.sum(np.square(Theta2[:, 1:])))

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


def feedforward(Theta1, Theta2, X_with_bias):
    """
    Execute feedforward propagation on all examples

    :param Theta1: a network parameter
    :param Theta2: a network parameter
    :param X_with_bias: a feature-bias matrix
    :return:
    """

    z2 = Theta1.dot(X_with_bias.T)
    a2 = np.c_[np.ones((dataset['X'].shape[0], 1)), sigmoid(z2).T]

    z3 = a2.dot(Theta2.T)
    a3 = sigmoid(z3)

    return a3


def gradientDescent(Initial_Theta1, Initial_Theta2, X_with_bias, y_one_hot, max_iterations, learning_rate,
                    lambda_param):
    Theta1 = Initial_Theta1
    Theta2 = Initial_Theta2

    for i in range(max_iterations):
        cost, gradient1, gradient2 = costFunction(Theta1, Theta2, X_with_bias, y_one_hot, lambda_param)

        print("Iteration {}: cost = {}".format(i, cost))

        Theta1 = Theta1 - (learning_rate * gradient1)
        Theta2 = Theta2 - (learning_rate * gradient2)

    return Theta1, Theta2


def randomizeInitialWeights(L_in, L_out):
    """
    We know that the parameter initialization should be randomly selected from the range
    [-\epsilon_init, \epsilon_init]. An idea to choose \epsilon_init is to base it on the
    number of units in the network. For example, ones can define \epsilon_init as follows:
    \epsilon_init = \sqrt{6} / (\sqrt{L_in + L_out}) where L_in = s_l and L_out = s_{l + 1}
    are the number of units in layers adjacent to \Theta^{(l)}

    :param L_in:
    :param L_out:
    :return:
    """
    initial_epsilon = (6 ** (1 / 2)) / ((L_in + L_out) ** (1 / 2))
    W = np.random.rand(L_out, L_in + 1) * 2 * initial_epsilon - initial_epsilon

    return W


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

X, y = dataset['X'], dataset['y']
print('X: {}'.format(X.shape))
print('y: {}'.format(y.shape))

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
Step 3: Test the implemented sigmoidGradient. 
For large values (both positive and negative), the gradient should be close to 0. 
Furthermore, when z = 0, the gradient should be exactly 0.25.
'''
print('Compute the sigmoidGradient for [-1, -0.5, 0, 0.5, 1] :', '\n',
      [sigmoidGradient(z) for z in [-1, -0.5, 0, 0.5, 1]], '\n')

'''
Step 4: Initialize the values of Theta1 and Theta2
'''

Initial_Theta1 = randomizeInitialWeights(400, 25)
Initial_Theta2 = randomizeInitialWeights(25, 10)

'''
Step 5: Train the neural network for 1,000 iterations using learning_rate = 0.8
'''

Theta1, Theta2 = gradientDescent(Initial_Theta1, Initial_Theta2, X_new, y_one_hot, 1000, 0.8, 0)

'''
Step 6: Make predictions. Noted that it is possible to get higher training accuracies by 
training the neural network for more iterations e.g. 3,000 iterations. 
'''

output = feedforward(Theta1, Theta2, X_new)

predicted_y = np.argmax(output, axis=1) + 1

# numpy.ravel() returns a contiguous flattened array.
print('Training set accuracy: {} %'.format(np.mean(predicted_y == y.ravel()) * 100))
