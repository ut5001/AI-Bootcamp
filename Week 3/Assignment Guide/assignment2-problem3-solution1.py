import tensorflow as tf
import numpy as np
import pandas as pd


def normalize(inputs):
    """
    Normalize an input array (feature scaling)

    Parameters
    ----------
    inputs : an input array

    Returns
    -------
    scaled_inputs : an input array in unit scale.
    """

    mean = np.mean(inputs)
    max_element = np.max(inputs)
    min_element = np.min(inputs)

    scaled_inputs = np.copy(inputs)

    for index in range(len(inputs)):
        scaled_inputs[index] = (inputs[index] - mean) / (max_element - min_element)

    return scaled_inputs


'''
Step 1: Read data from CSV file using Pandas
'''
data_from_CSV = pd.read_csv("data/boston.csv")

feature_CRIM = data_from_CSV['CRIM']
feature_ZN = data_from_CSV['ZN']
feature_INDUS = data_from_CSV['INDUS']
feature_CHAS = data_from_CSV['CHAS']
feature_NOX = data_from_CSV['NOX']
feature_RM = data_from_CSV['RM']
feature_AGE = data_from_CSV['AGE']
feature_DIS = data_from_CSV['DIS']
feature_RAD = data_from_CSV['RAD']
feature_TAX = data_from_CSV['TAX']
feature_PTRATIO = data_from_CSV['PTRATIO']
feature_LSTAT = data_from_CSV['LSTAT']

target_MEDV = data_from_CSV['MEDV']

'''
Step 2: Rescale the training dataset
'''
scaled_feature_CRIM = normalize(feature_CRIM)
scaled_feature_ZN = normalize(feature_ZN)
scaled_feature_INDUS = normalize(feature_INDUS)
scaled_feature_CHAS = normalize(feature_CHAS)
scaled_feature_NOX = normalize(feature_NOX)
scaled_feature_RM = normalize(feature_RM)
scaled_feature_AGE = normalize(feature_AGE)
scaled_feature_DIS = normalize(feature_DIS)
scaled_feature_RAD = normalize(feature_RAD)
scaled_feature_TAX = normalize(feature_TAX)
scaled_feature_PTRATIO = normalize(feature_PTRATIO)
scaled_feature_LSTAT = normalize(feature_LSTAT)

'''
Step 3: Create placeholders for features Xs and target Y
'''
X_CRIM = tf.placeholder(tf.float32, name='X_CRIM')
X_ZN = tf.placeholder(tf.float32, name='X_ZN')
X_INDUS = tf.placeholder(tf.float32, name='X_INDUS')
X_CHAS = tf.placeholder(tf.float32, name='X_CHAS')
X_NOX = tf.placeholder(tf.float32, name='X_NOX')
X_RM = tf.placeholder(tf.float32, name='X_RM')
X_AGE = tf.placeholder(tf.float32, name='X_AGE')
X_DIS = tf.placeholder(tf.float32, name='X_DIS')
X_RAD = tf.placeholder(tf.float32, name='X_RAD')
X_TAX = tf.placeholder(tf.float32, name='X_TAX')
X_PTRATIO = tf.placeholder(tf.float32, name='X_PTRATIO')
X_LSTAT = tf.placeholder(tf.float32, name='X_LSTAT')

Y = tf.placeholder(tf.float32, name='Y')

'''
Step 4: Create thetas, initialized them to 0
'''
theta0 = tf.Variable(0.0, name='theta0')
theta_CRIM = tf.Variable(0.0, name='theta_CRIM')
theta_ZN = tf.Variable(0.0, name='theta_ZN')
theta_INDUS = tf.Variable(0.0, name='theta_INDUS')
theta_CHAS = tf.Variable(0.0, name='theta_CHAS')
theta_NOX = tf.Variable(0.0, name='theta_NOX')
theta_RM = tf.Variable(0.0, name='theta_RM')
theta_AGE = tf.Variable(0.0, name='theta_AGE')
theta_DIS = tf.Variable(0.0, name='theta_DIS')
theta_RAD = tf.Variable(0.0, name='theta_RAD')
theta_TAX = tf.Variable(0.0, name='theta_TAX')
theta_PTRATIO = tf.Variable(0.0, name='theta_PTRATIO')
theta_LSTAT = tf.Variable(0.0, name='theta_LSTAT')

'''
Step 5: Define a hypothesis function to predict Y
'''
hypothesis_function = theta0 + theta_CRIM * X_CRIM + theta_ZN * X_ZN + theta_INDUS * X_INDUS + \
                      theta_CHAS * X_CHAS + theta_NOX * X_NOX + theta_RM * X_RM + theta_AGE * X_AGE + \
                      theta_DIS * X_DIS + theta_RAD * X_RAD + theta_TAX * X_TAX + theta_PTRATIO * X_PTRATIO + \
                      theta_LSTAT * X_LSTAT

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
    Step 8: Initialize the necessary variables
    '''
    session.run(tf.global_variables_initializer())

    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter('./graphs/multivariate_linear_regression_feature_scaling', session.graph)

    '''
    Step 9: Train the model for 1,000 epochs
    '''
    for i in range(1000):
        summary, _, cost = session.run([merged, optimizer, cost_function],
                                       feed_dict={X_CRIM: scaled_feature_CRIM, X_ZN: scaled_feature_ZN,
                                          X_INDUS: scaled_feature_INDUS, X_CHAS: scaled_feature_CHAS,
                                          X_NOX: scaled_feature_NOX, X_RM: scaled_feature_RM,
                                          X_AGE: scaled_feature_AGE, X_DIS: scaled_feature_DIS,
                                          X_RAD: scaled_feature_RAD, X_TAX: scaled_feature_TAX,
                                          X_PTRATIO: scaled_feature_PTRATIO, X_LSTAT: scaled_feature_LSTAT,
                                          Y: target_MEDV})

        writer.add_summary(summary, i)

        print("Epoch: {0}, cost = {1}".format(i+1, cost))

    '''
    Step 10: Prints the training cost and all thetas
    '''
    print("Optimization Finished!", '\n')
    print("Training cost = {}".format(cost))
    print("theta0 = {}".format(session.run(theta0)))
    print("theta_CRIM = {}".format(session.run(theta_CRIM)))
    print("theta_ZN = {}".format(session.run(theta_ZN)))
    print("theta_INDUS = {}".format(session.run(theta_INDUS)))
    print("theta_CHAS = {}".format(session.run(theta_CHAS)))
    print("theta_NOX = {}".format(session.run(theta_NOX)))
    print("theta_RM = {}".format(session.run(theta_RM)))
    print("theta_AGE = {}".format(session.run(theta_AGE)))
    print("theta_DIS = {}".format(session.run(theta_DIS)))
    print("theta_RAD = {}".format(session.run(theta_RAD)))
    print("theta_TAX = {}".format(session.run(theta_TAX)))
    print("theta_PTRATIO = {}".format(session.run(theta_PTRATIO)))
    print("theta_LSTAT = {}".format(session.run(theta_LSTAT)))


# Close the writer when you finished using it
writer.close()


