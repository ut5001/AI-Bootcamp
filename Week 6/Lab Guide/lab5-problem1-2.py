import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
from scipy.optimize import minimize


def costFunction_regularized_logistic_regression(theta_param, lambda_param, X, y):
    number_of_examples = X.shape[0]
    predicted_y = sigmoid(X.dot(theta_param))

    cost = -1 * (1 / number_of_examples) * (np.log(predicted_y).T.dot(y) + np.log(1 - predicted_y).T.dot(1 - y)) \
           + (lambda_param / (2 * number_of_examples)) * np.sum(np.square(theta_param[1:]))

    if np.isnan(cost):
        return np.inf
    return cost


def gradient_regularized_logistic_regression(theta_param, lambda_param, X, y):
    number_of_examples = X.shape[0]
    # -1 simply means unknown and we want numpy to figure out for us
    predicted_y = sigmoid(X.dot(theta_param.reshape(-1, 1)))

    gradient = (1 / number_of_examples) * X.T.dot(predicted_y - y) \
           + (lambda_param / number_of_examples) * np.r_[[[0]], theta_param[1:].reshape(-1, 1)]

    return gradient.flatten()


def mapFeature(X, degree):
    polynomial_features = PolynomialFeatures(degree)

    return polynomial_features, polynomial_features.fit_transform(X)


def plotData(data, label_x, label_y, label_pos, label_neg, axes=None):
    # Get indexes for class 0 and class 1
    neg = data[:, 2] == 0
    pos = data[:, 2] == 1

    # If no specific axes object has been passed, get the current axes.
    if axes is None:
        axes = plt.gca()

    axes.scatter(data[pos][:, 0], data[pos][:, 1], marker='o', label=label_pos)
    axes.scatter(data[neg][:, 0], data[neg][:, 1], marker='x', label=label_neg)
    axes.set_xlabel(label_x)
    axes.set_ylabel(label_y)
    axes.legend(frameon=True, fancybox=True)


def sigmoid(z):
    return 1.0 / (1 + np.exp(-z))


dataframe = pd.read_csv('data/problem1data.txt', header=None)
y = dataframe[2].values
X = dataframe[[0, 1]].values

polynomial_features, X_New = mapFeature(X, 6)

initial_theta = np.zeros(X_New.shape[1])
print('Cost at the initial theta is ', costFunction_regularized_logistic_regression(initial_theta, 1, X_New, y))
print('Gradient at the initial theta is ', gradient_regularized_logistic_regression(initial_theta, 0, X_New, y))

fig, axes = plt.subplots(1, 3, sharey=True, figsize=(17, 5))

for i, lambda_param in enumerate([0, 1, 100]):
    optimal_param = minimize(costFunction_regularized_logistic_regression, initial_theta,
                             (lambda_param, X_New, y.reshape(-1, 1)), method=None,
                             jac=gradient_regularized_logistic_regression,
                             options={'maxiter': 3000})

    plotData(dataframe.values, 'Microchip Test 1', 'Microchip Test 2', 'y = 1', 'y = 0', axes.flatten()[i])
    plt.xlabel('Microchip Test 1')
    plt.ylabel('Microchip Test 1')
    plt.title('Plot of training data')
    plt.legend()
    plt.show()

    # Plot decisionboundary
    x1_min, x1_max = X[:, 0].min(), X[:, 0].max(),
    x2_min, x2_max = X[:, 1].min(), X[:, 1].max(),
    xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max), np.linspace(x2_min, x2_max))

    predicted_y = sigmoid(polynomial_features.fit_transform(np.c_[xx1.ravel(), xx2.ravel()]).dot(optimal_param.x))
    predicted_y = predicted_y.reshape(xx1.shape)
    axes.flatten()[i].contour(xx1, xx2, predicted_y, [0.5], linewidths=1, colors='g');
    axes.flatten()[i].set_title('Lambda = {}'.format(lambda_param))
