import numpy as np
import scipy.io as sio
import tensorflow as tf


def one_hot_vector(i):
    one_hot = np.zeros(10, 'uint8')
    one_hot[i] = 1
    return one_hot


'''
Step 0: Load the input data, which is stored in digits.mat file. 
Then, initialize some variables namely number_of_samples, number_of_features, and number_of_labels
'''

data = sio.loadmat("data/digits.mat")

feature_matrix = data['X']  # the feature matrix is labeled with 'X' inside the file
target_vector = np.squeeze(data['y'])  # the target variable vector is labeled with 'y' inside the file

number_of_samples = feature_matrix.shape[0]  # i.e. 5000 samples
number_of_features = feature_matrix.shape[1]  # i.e. 400 features (20 * 20)
number_of_labels = np.max(target_vector)  # i.e. 1, 2, 3, ..., 10

target_vector_one_hot = np.zeros([number_of_samples, number_of_labels])
for i in range(number_of_samples):
    target_vector_one_hot[i, :] = one_hot_vector(target_vector[i] % 10)

'''
Step 1: Construct TF graph
'''

X = tf.placeholder(tf.float32, [None, number_of_features], name='X')
Y = tf.placeholder(tf.float32, [None, 10], name='Y')  # 0-9 digits recognition. Hence, 10 classes !

thetas = tf.Variable(tf.zeros([number_of_features, 10]), name='Thetas')
theta0 = tf.Variable(tf.zeros([10]), name='theta0')

'''
Step 2: Define the hypothesis function i.e. Softmax function (see Slide# )
'''
z = tf.matmul(X, thetas)
hypothesis_function = tf.nn.softmax(z + theta0)

'''
Step 3: Define the cost function i.e. the cross-entropy loss
'''

# Why reduction_indices = 1?
# Y * tf.log(hypothesis_function) will produce the class probabilities for each training example,
# where each row represent an example and each column represent its probability of belonging in a certain class.
# The idea of cross-entropy loss is to compute the error between predicted classes and its actual class.
# Since each instance may be belonged to multiple classes according to different probability,
# so we need to sum up along columns i.e. reduction_indices = 1.
# Noted that -tf.reduce_sum(Y * tf.log(hypothesis_function), reduction_indices=1) will produce
# something like [2.3025851 2.3025851 2.3025851 ... 2.3025851 2.3025851 2.3025851] (its length is 5,000)
# Finally, we want to find the total loss over all examples. That's why we compute the average.
# That's the idea of the cross-entropy loss !
cost_function = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis_function), reduction_indices=1))

'''
Step 4: Use gradient descent with learning rate of 0.9 to minimize the cost function
'''

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.9).minimize(cost_function)

'''
Step 5: Train the model 
'''

with tf.Session() as session:
    session.run(tf.global_variables_initializer())

    for i in range(1000):
        _, cost = session.run([optimizer, cost_function], feed_dict={X: feature_matrix, Y: target_vector_one_hot})

        print("Epoch = {0}, Cost = {1}\n".format(i + 1, cost))

    print("Optimization Finished!")
    # Test model
    correct_prediction = tf.equal(tf.argmax(hypothesis_function, 1), tf.argmax(Y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print("Accuracy:", accuracy.eval({X: feature_matrix, Y: target_vector_one_hot}))

