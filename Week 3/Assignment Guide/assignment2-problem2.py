import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import xlrd

'''
Step 0: Suppress warning logs
'''
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

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
Step 2: Rescale the training dataset
'''
average_of_X = np.mean(data.T[0])
max_of_X = np.max(data.T[0])
min_of_X = np.min(data.T[0])

scaled_X = np.copy(data.T[0])
for i in range(len(data.T[0])):
    scaled_X[i] = (data.T[0][i] - average_of_X)/(max_of_X - min_of_X)

'''
Step 3: Create placeholders for feature X (number of fire) and target Y (number of theft)
'''
X = tf.placeholder(tf.float32, name='X')
Y = tf.placeholder(tf.float32, name='Y')

'''
Step 4: Create theta0, theta1 and theta2; then, initialized them to 0
'''
theta0 = tf.Variable(0.0, name='theta0')
theta1 = tf.Variable(0.0, name='theta1')
theta2 = tf.Variable(0.0, name='theta2')

'''
Step 5: Define a hypothesis function to predict Y
'''
hypothesis_function = theta0 + (theta1 * X) + (theta2 * X * X)

'''
Step 6: Use the square error as the cost function
'''
cost_function = tf.multiply(tf.divide(1, 2), tf.reduce_mean(tf.pow(Y - hypothesis_function, 2)))
tf.summary.scalar('total cost', cost_function)

'''
Step 7: Using gradient descent with learning rate of 0.3 to minimize cost
'''
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.3).minimize(cost_function)

with tf.Session() as session:
    '''
    Step 8: Initialize the necessary variables, i.e. theta0 and theta1
    '''
    session.run(tf.global_variables_initializer())

    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter('./graphs/polynomial_regression_feature_scaling', session.graph)

    '''
    Step 9: Train the model for 30,000 epochs
    '''
    for i in range(30000):
        session.run(optimizer, feed_dict={X: scaled_X, Y: data.T[1]})

        summary, cost = session.run([merged, cost_function], feed_dict={X: scaled_X, Y: data.T[1]})
        writer.add_summary(summary, i)

        print("Epoch: {0}, cost = {1}, theta0 = {2}, theta1 = {3}, theta2 = {4}".
              format(i + 1, cost, session.run(theta0), session.run(theta1), session.run(theta1)))

    '''
    Step 10: Prints the training cost, theta0, and theta1
    '''
    print("Optimization Finished!")
    training_cost = session.run(cost_function, feed_dict={X: scaled_X, Y: data.T[1]})
    print("Training cost =", training_cost, "theta0 = ", session.run(theta0), "theta1 = ", session.run(theta1),
          "theta2 = ", session.run(theta2), '\n')

    '''
    Step 11: Plot the results
    '''
    # Graphic display
    plt.plot(scaled_X, data.T[1], 'ro', label='Original data')
    plt.plot(scaled_X, session.run(theta0) + (session.run(theta1) * scaled_X) + (session.run(theta2) * scaled_X * scaled_X),
             'bo', label='Fitted line')
    plt.xlabel('fire per 1000 housing units')
    plt.ylabel('theft per 1000 population')
    plt.legend()
    plt.show()

# Close the writer when you finished using it
writer.close()
