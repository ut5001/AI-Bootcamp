import numpy as np
import matplotlib.pyplot as plt
import xlrd


def gradient_descent(X, Y, alpha=0.01, converge_criteria=0.0001, max_epoch=10000):
    converged = False
    epoch = 0

    # Number of datasets
    m = X.shape[0]

    # Initial thetas to 0
    theta0 = 0
    theta1 = 0

    # Total error, J(theta)
    J = (1.0 / (2.0 * m)) * sum([ (theta0 + theta1 * X[i] - Y[i]) ** 2 for i in range(m)])

    while not converged:
        # For each training sample, compute the gradient
        gradient0 = (1.0 / m) * sum([(theta0 + theta1 * X[i] - Y[i]) for i in range(m)])
        gradient1 = (1.0 / m) * sum([(theta0 + theta1 * X[i] - Y[i]) * X[i] for i in range(m)])

        # Update the temporary thetas
        tmp0 = theta0 - alpha * gradient0
        tmp1 = theta1 - alpha * gradient1

        # Update thetas
        theta0 = tmp0
        theta1 = tmp1

        # Mean squared error
        error = (1.0 / (2.0 * m)) * sum([(theta0 + theta1 * X[i] - Y[i]) ** 2 for i in range(m)])

        print("Epoch = {0}, Cost = {1}".format(epoch, error.item(0)))

        if abs(J - error) <= converge_criteria:
            print('Converged, epochs: ', epoch, '!!!')
            converged = True

        J = error  # Update error
        epoch += 1  # Update epoch

        if epoch == max_epoch:
            print('Maximum number of epochs exceeded!')
            converged = True

    return theta0.item(0), theta1.item(0)


''' 
Step 1: Read in data from the .xls file
'''
DATA_FILE = 'data/fire_theft.xls'

book = xlrd.open_workbook(DATA_FILE, encoding_override='utf-8')
sheet = book.sheet_by_index(0)

number_of_rows = len(list(sheet.get_rows()))
data = np.asarray([sheet.row_values(i) for i in range(1, number_of_rows)])
number_of_samples = number_of_rows - 1

'''
Step 2: Compute the gradients
'''
X, Y = np.matrix(data.T[0]).T, np.matrix(data.T[1]).T

theta0, theta1 = gradient_descent(X, Y, alpha=0.001,
                                  converge_criteria=0.0000001, max_epoch=300000)

print("theta0 = {0}, theta1 = {1}".format(theta0, theta1))

'''
Step 3: Plot the results
'''

# Graphic display
plt.plot(data.T[0], data.T[1], 'ro', label='Original data')
plt.plot(data.T[0], theta0 + theta1 * data.T[0], 'b', label='Fitted line')
plt.xlabel('fire per 1000 housing units')
plt.ylabel('theft per 1000 population')
plt.legend()
plt.show()

