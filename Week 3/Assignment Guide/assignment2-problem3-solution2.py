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
Step 2: Rescale the training dataset; then, construct the features and target matrix
'''
scaled_feature_CRIM = np.matrix(normalize(feature_CRIM)).T
scaled_feature_ZN = np.matrix(normalize(feature_ZN)).T
scaled_feature_INDUS = np.matrix(normalize(feature_INDUS)).T
scaled_feature_CHAS = np.matrix(normalize(feature_CHAS)).T
scaled_feature_NOX = np.matrix(normalize(feature_NOX)).T
scaled_feature_RM = np.matrix(normalize(feature_RM)).T
scaled_feature_AGE = np.matrix(normalize(feature_AGE)).T
scaled_feature_DIS = np.matrix(normalize(feature_DIS)).T
scaled_feature_RAD = np.matrix(normalize(feature_RAD)).T
scaled_feature_TAX = np.matrix(normalize(feature_TAX)).T
scaled_feature_PTRATIO = np.matrix(normalize(feature_PTRATIO)).T
scaled_feature_LSTAT = np.matrix(normalize(feature_LSTAT)).T

features_matrix = np.concatenate((scaled_feature_CRIM, scaled_feature_ZN, scaled_feature_INDUS, scaled_feature_CHAS,
                            scaled_feature_NOX, scaled_feature_RM, scaled_feature_AGE, scaled_feature_DIS, scaled_feature_RAD,
                            scaled_feature_TAX, scaled_feature_PTRATIO, scaled_feature_LSTAT), axis=1)

target_matrix = np.matrix(target_MEDV).T

'''
Step 3: Create placeholders for features Xs and target Y
'''

# When the summation of (theta_i)(x_i) is huge, we can utilize the
# matrix multiplication. In order to do this, we declare placeholders as 2-D matrices.
# In the following, [None, 12] means any number of rows and 12 columns.
X = tf.placeholder(tf.float32, [None, 12], name='X')
Y = tf.placeholder(tf.float32, [None, 1], name='Y')

'''
Step 4: Create thetas, initialized them to 0
'''
thetas = tf.Variable(tf.zeros([12, 1]), name='Thetas')
theta0 = tf.Variable(tf.zeros([1]), name='Theta0')

'''
Step 5: Define a hypothesis function to predict Y
'''

# We can tell TensorBoard to group a certain set of nodes together.
# In order to this, we use "with tf.name_scope(....) as scope".
# Try to run TensorBoard and see what happens on the graph tab.
with tf.name_scope('Hypothesis_Function') as scope:
    # Recall that (Theta_i)(X_i) = (theta_1)(x_1) + ... + (theta_i)(x_i) + ... + (theta_n)(x_n)
    # Noted that I use capital letters to denote matrices.
    feature_theta_multiplication = tf.matmul(X, thetas)
    hypothesis_function = feature_theta_multiplication + theta0

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
                                       feed_dict={X: features_matrix, Y: target_matrix})

        writer.add_summary(summary, i)

        print("Epoch: {0}, cost = {1}".format(i+1, cost))

    '''
    Step 10: Prints the training cost and all thetas
    '''
    print("Optimization Finished!", '\n')
    print("Training cost = {}".format(cost))
    print("theta0 = {}".format(session.run(theta0)[0]))
    print("theta_CRIM = {}".format(session.run(thetas)[0][0]))
    print("theta_ZN = {}".format(session.run(thetas)[1][0]))
    print("theta_INDUS = {}".format(session.run(thetas)[2][0]))
    print("theta_CHAS = {}".format(session.run(thetas)[3][0]))
    print("theta_NOX = {}".format(session.run(thetas)[4][0]))
    print("theta_RM = {}".format(session.run(thetas)[5][0]))
    print("theta_AGE = {}".format(session.run(thetas)[6][0]))
    print("theta_DIS = {}".format(session.run(thetas)[7][0]))
    print("theta_RAD = {}".format(session.run(thetas)[8][0]))
    print("theta_TAX = {}".format(session.run(thetas)[9][0]))
    print("theta_PTRATIO = {}".format(session.run(thetas)[10][0]))
    print("theta_LSTAT = {}".format(session.run(thetas)[11][0]))

# Close the writer when you finished using it
writer.close()


