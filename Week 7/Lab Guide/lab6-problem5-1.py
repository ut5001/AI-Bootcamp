import numpy as np
from sklearn.datasets import load_breast_cancer
import tensorflow as tf


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
label = np.matrix(dataset.target).T
Input = np.matrix(dataset.data)

number_examples = Input.shape[0]
number_features = Input.shape[1]

# Normalize each input feature
Input_new = normalize(Input)

print("Dimension of X (normalized): {}".format(Input_new.shape))
print("Dimension of y: {}".format(label.shape))
print('\n')


'''
Step 1: Set up basic parameters
'''

learning_rate = 0.001
training_steps = 2
batch_size = 10

# Network parameters

neurons_hidden1 = 5
neurons_hidden2 = 6
neurons_input = number_features
neurons_output = 1

'''
Step 2: Set up TF Graph's input
'''

X = tf.placeholder("float", [None, neurons_input])
Y = tf.placeholder("float", [None, neurons_output])


'''
Step 3: Set up weights & biases of each layer
'''

weights = {
    'hidden1': tf.Variable(tf.random_normal([neurons_input, neurons_hidden1])),
    'hidden2': tf.Variable(tf.random_normal([neurons_hidden1, neurons_hidden2])),
    'output': tf.Variable(tf.random_normal([neurons_hidden2, neurons_output]))
}

biases = {
    'bias1': tf.Variable(tf.random_normal([neurons_hidden1])),
    'bias2': tf.Variable(tf.random_normal([neurons_hidden2])),
    'output': tf.Variable(tf.random_normal([neurons_output]))
}


'''
Step 4: Create a neural network's architecture (sometimes, this may be called 'model').
'''

Hidden1 = tf.add(tf.matmul(X, weights['hidden1']), biases['bias1'])
Hidden2 = tf.add(tf.matmul(Hidden1, weights['hidden2']), biases['bias2'])
Output = tf.add(tf.matmul(Hidden2, weights['output']), biases['output'])

hypothesis_function = tf.nn.softmax(Output)


'''
Step 5: Define a cost function and its optimizer
'''

cost_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=hypothesis_function, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

training_operation = optimizer.minimize(cost_function)

'''
Step 6: Define evaluation model
'''

number_of_correct_prediction = tf.equal(tf.argmax(hypothesis_function, axis=1), tf.argmax(Y, axis=1))
accuracy_function = tf.reduce_mean(tf.cast(number_of_correct_prediction, tf.float32))

'''
Step 7: Start training the neural network
'''

# Start training in a session
with tf.Session() as session:

    # Run the initializer
    session.run(tf.global_variables_initializer())

    for i in range(1, training_steps + 1):

        batch_number = int(number_examples / batch_size)
        order = np.random.permutation(number_examples)

        # Run optimization op (backprop)
        for j in range(batch_number):

            print("Index {} - {}".format(j * batch_size, (j+1) * batch_size - 1))
            session.run(training_operation, feed_dict={X: Input_new[order[j * batch_size]:order[(j+1) * batch_size - 1]],
                                                       Y: label[order[j * batch_size]:order[(j+1) * batch_size - 1]]})
            cost, accuracy = session.run([cost_function, accuracy_function],
                                         feed_dict={X: Input_new[order[j * batch_size]:order[(j+1) * batch_size - 1]],
                                                    Y: label[order[j * batch_size]:order[(j+1) * batch_size - 1]]})
            print("Epoch {} Iteration {}: Cost = {}, Accuracy = {}".format(i, j, cost, accuracy))

        print('\n')

    print("Optimization Finished!")

    cost, accuracy = session.run([cost_function, accuracy_function], feed_dict={X: Input_new, Y: label})
    print("Cost = {}, Accuracy = {}".format(cost, accuracy))

