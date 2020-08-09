import numpy as np
from scipy.io import loadmat # for loading MATLAB files
import matplotlib.pyplot as plt


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
print('X_new (with intercept): {}'.format(X_new.shape))


'''
Step 2: Now, we want to visualize some examples based on the following steps:
1. sample 20 images randomly from matrix X_new
2. visualize them using method 'imshow' in matplot
'''

sample = np.random.choice(X_new.shape[0], 20)

# X_new[sample,1:].reshape(-1,20) is a matrix of dimension 400 x 20
plt.imshow(X_new[sample,1:].reshape(-1,20).T)
plt.axis('off');


'''
Step 3: Call the feedforward to obtain a matrix output. Then, use it for the predictions.
'''

output = feedforward(Theta1, Theta2, X_new)

predicted_y = np.argmax(output, axis=1) + 1

# numpy.ravel() returns a contiguous flattened array.
print('Training set accuracy: {} %'.format(np.mean(predicted_y == y.ravel()) * 100))
