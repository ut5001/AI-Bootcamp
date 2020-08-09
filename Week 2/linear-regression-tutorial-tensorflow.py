import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

'''
Step 1: Define training dataset
'''

area = np.asarray([3.3, 4.4, 5.5, 6.71, 6.93, 4.168, 9.779, 6.182, 7.59, 2.167, 7.042, 10.791, 5.313, 7.997, 5.654, 9.27, 3.1])
price = np.asarray([1.7, 2.76, 2.09, 3.19, 1.694, 1.573, 3.366, 2.596, 2.53, 1.221, 2.827, 3.465, 1.65, 2.904, 2.42, 2.94, 1.3])

n_samples = area.shape[0]

'''
Step 2: Create placeholders for training X and label Y
'''
X = tf.placeholder(tf.float32, name='X')
Y = tf.placeholder(tf.float32, name='Y')


'''
Step 3: Create theta0 and theta1, initialized them to 0
'''
theta0 = tf.Variable(0.0, name='theta0')
theta1 = tf.Variable(0.0, name='theta1')


'''
Step 4: Define a hypothesis function to predict Y
'''
hypothesis_function = theta0 + theta1 * X


'''
Step 5: Use the mean squared error as the loss function
'''
loss_function = (1 / (2 * n_samples)) * tf.reduce_sum(tf.pow(Y - hypothesis_function, 2))


'''
Step 6: Using gradient descent with learning rate of 0.03 to minimize the loss
'''

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.03).minimize(loss_function)

with tf.Session() as session:
    '''
    Step 7: Initialize the necessary variables, i.e. theta0 and theta1
    '''
    session.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter('./graphs/linear_regression', session.graph)

    '''
    Step 8: Train the models for 1000 epoches
    '''
    for i in range(1000):
        session.run(optimizer, feed_dict={X:area, Y: price})
        cost = session.run(loss_function, feed_dict={X: area, Y:price})
        
        print("Epoch: {0}, cost = {1}, theta0 = {2}, theta1 = {3}".format(i + 1, cost, session.run(theta0), session.run(theta1)))

    '''
    Step 9: Prints the training cost, theta0, and theta1
    '''
    print("Optimization Finished!")
    training_cost = session.run(loss_function, feed_dict={X: area, Y: price})
    print("Training cost =", training_cost, "theta0 = ", session.run(theta0), "theta1 = ", session.run(theta1), '\n')

    '''
    Step 10: Plot the results
    '''
    # Graphic display
    plt.plot(area, price, 'ro', label='Original data')
    plt.plot(area, session.run(theta1) * area + session.run(theta0), 'b', label='Fitted line')
    plt.xlabel('Area (in 10 m^2)')
    plt.ylabel('Price (in MB)')
    plt.legend()
    plt.show()

# Close the writer when you finished using it
writer.close()