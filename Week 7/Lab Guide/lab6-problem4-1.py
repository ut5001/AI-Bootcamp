import numpy as np
from sklearn.datasets import load_breast_cancer


def activation(z):
    return 1 / (1 + np.exp(-z))


def activationDerivative(z):
    tmp = activation(z)

    return np.multiply(tmp, 1 - tmp)


def feedforward(x, W, b):

    L = len(W) - 1
    a = x

    for l in range(1, L + 1):
        z = W[l].T * a + b[l]
        a = activation(z)

    return a


def costFunction(y, predicted_y):
    return -((1 - y) * np.log(1 - predicted_y) + y * np.log(predicted_y))


def normalize(X):
    """
    Normalize each input feature

    :param X:
    :return:
    """

    number_examples = X.shape[0]

    X_normalized = X - np.tile(np.mean(X, 0), [number_examples, 1])
    X_normalized = np.divide(X_normalized, np.tile(np.std(X_normalized, 0), [number_examples, 1]))

    return X_normalized


'''
Step 0: Load the training dataset
'''

dataset = load_breast_cancer()
y = np.matrix(dataset.target).T
X = np.matrix(dataset.data)

number_examples = X.shape[0]
number_features = X.shape[1]

# Normalize each input feature
X_new = normalize(X)

print("Dimension of X (normalized): {}".format(X_new.shape))
print("Dimension of y: {}".format(y.shape))
print('\n')

'''
Step 1: Construct a neural network as follows:
- 4 layers with sigmoid activation functions
- 6 units in 1st hidden layer
- 5 units in 2nd hidden layer
'''

hidden2 = 5
hidden1 = 6

Weights = [[], np.random.normal(0, 0.1, [number_features, hidden1]),
           np.random.normal(0, 0.1, [hidden1, hidden2]),
           np.random.normal(0, 0.1, [hidden2, 1])]
Biases = [[], np.random.normal(0, 0.1, [hidden1, 1]),
          np.random.normal(0, 0.1, [hidden2, 1]),
          np.random.normal(0, 0.1, [1, 1])]

L = len(Weights) - 1


'''
Step 2: Implement the stochastic gradient descent with:
- learning rate = 0.01
- max iteration = 1,000
'''

learning_rate = 0.01
max_iteration = 1000

for iteration in range(0, max_iteration):
    loss_this_iter = 0
    order = np.random.permutation(number_examples)

    for i in range(0, number_examples):

        # Grab the pattern order[i]
        x_sample = X_new[order[i], :].T
        y_sample = y[order[i], 0]

        # Feed forward step
        a_arr = [x_sample]
        z_arr = [[]]
        delta = [[]]

        gradient_weights = [[]]
        gradient_biases = [[]]

        for l in range(1, L + 1):
            z_arr.append(Weights[l].T * a_arr[l - 1] + Biases[l])
            a_arr.append(activation(z_arr[l]))

            # Just to give arrays the right shape for the backprop step
            delta.append([])
            gradient_weights.append([])
            gradient_biases.append([])

        loss_this_sample = costFunction(y_sample, a_arr[L][0, 0])
        loss_this_iter = loss_this_iter + loss_this_sample

        # Backprop step
        delta[L] = a_arr[L] - y_sample

        for l in range(L, 0, -1):
            gradient_biases[l] = delta[l].copy()
            gradient_weights[l] = a_arr[l - 1] * delta[l].T

            if l > 1:
                delta[l - 1] = np.multiply(activationDerivative(z_arr[l - 1]), Weights[l] * delta[l])

        # Gradient checking
        if False:
            print('\n')
            print("Target: {}".format(y_sample))
            print("Predicted y: {}".format(a_arr[L][0, 0]))

            diff = 1e-4
            Weights[1][10, 0] = Weights[1][10, 0] + diff
            predicted_y_analytic_positive = feedforward(x_sample, Weights, Biases)

            Weights[1][10, 0] = Weights[1][10, 0] - (2 * diff)
            predicted_y_analytic_negative = feedforward(x_sample, Weights, Biases)

            cost_positive = costFunction(y_sample, predicted_y_analytic_positive)
            cost_negative = costFunction(y_sample, predicted_y_analytic_negative)

            gradient_analytic = (cost_positive - cost_negative) / (2 * diff)

            print("Positive Cost: {}, Negative Cost: {}".format(cost_positive[0, 0], cost_negative[0, 0]))
            print("Theoretical Gradient: {}, Analytical Gradient: {}".format(gradient_weights[1][10, 0],
                                                                             gradient_analytic[0, 0]))

        for l in range(1, L + 1):
            Weights[l] = Weights[l] - learning_rate * gradient_weights[l]
            Biases[l] = Biases[l] - learning_rate * gradient_biases[l]

    print('\n')
    print("Iteration {} loss {}".format(iteration, loss_this_iter))
