import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.datasets
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.python.framework import ops


def create_placeholders(n_x, n_y):

    X = tf.placeholder("float", [n_x, None])
    Y = tf.placeholder("float", [n_y, None])

    return X, Y

def initialize_parameters(layers_dims):

    tf.set_random_seed(2)

    parameters = {}
    L = len(layers_dims) - 1

    for l in range(1, L + 1):

        parameters['W' + str(l)] = tf.get_variable("W"+str(l), [layers_dims[l], layers_dims[l-1]], initializer=tf.contrib.layers.xavier_initializer(seed=1))
        parameters['b' + str(l)] = tf.get_variable("b"+str(l), [layers_dims[l], 1], initializer=tf.zeros_initializer())

    return parameters

def forward_propagation(X, parameters):

    L = len(parameters) // 2
    A = X

    for l in range(1, L):
        A_prev = A

        Z = tf.add(tf.matmul(parameters['W'+str(l)], A_prev), parameters['b'+str(l)])
        A = tf.nn.relu(Z)

    ZL = tf.add(tf.matmul(parameters['W'+str(L)], A), parameters['b'+str(L)])

    return ZL

def compute_cost(ZL, Y):

    logits = tf.transpose(ZL)
    labels = tf.transpose(Y)

    cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels))

    return cost

def random_mini_batches(X, Y, mini_batch_size = 64, seed = 0):

    m = X.shape[1]
    mini_batches = []
    np.random.seed(seed)

    # Shuffle the data set
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation].reshape((Y.shape[0],m))

    # Partition the data set, except for the last partition
    num_complete_minibatches = math.floor(m/mini_batch_size)
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[:, k * mini_batch_size : k * mini_batch_size + mini_batch_size]
        mini_batch_Y = shuffled_Y[:, k * mini_batch_size : k * mini_batch_size + mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    # The last batch
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[:, num_complete_minibatches * mini_batch_size : m]
        mini_batch_Y = shuffled_Y[:, num_complete_minibatches * mini_batch_size : m]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    return mini_batches

def model(X_train, Y_train, X_test, Y_test, layers_dims, learning_rate = 0.0005,
          num_epochs = 700, minibatch_size = 32, print_cost = True, seed=66):

    ops.reset_default_graph()
    tf.set_random_seed(1)
    seed = seed
    (n_x, m) = X_train.shape
    n_y = Y_train.shape[0]
    costs = []

    X, Y = create_placeholders(n_x, n_y)

    parameters = initialize_parameters(layers_dims)

    ZL = forward_propagation(X, parameters)

    cost = compute_cost(ZL, Y)

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:

        sess.run(init)

        for epoch in range(num_epochs):

            epoch_cost = 0.
            num_minibatches = int(m / minibatch_size)
            seed = seed + 1
            minibatches = random_mini_batches(X_train, Y_train, minibatch_size, seed)

            for minibatch in minibatches:

                (minibatch_X, minibatch_Y) = minibatch

                _ , minibatch_cost = sess.run([optimizer, cost],
                                             feed_dict={X: minibatch_X,
                                                        Y: minibatch_Y})

                epoch_cost += minibatch_cost / num_minibatches

            if print_cost == True and epoch % 100 == 0:
                print ("Cost after epoch %i: %f" % (epoch, epoch_cost))
            if print_cost == True and epoch % 5 == 0:
                costs.append(epoch_cost)

        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()

        parameters = sess.run(parameters)

        correct_prediction = tf.equal(tf.argmax(ZL), tf.argmax(Y))

        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        print ("Train Accuracy:", accuracy.eval({X: X_train, Y: Y_train}))
        print ("Test Accuracy:", accuracy.eval({X: X_test, Y: Y_test}))

        return parameters

def predict(X, parameters):
    '''
    shape of X is (nx, m)
    '''
    L = len(parameters) // 2
    params = {}

    for l in range(L):
        params['W'+str(l+1)] = tf.convert_to_tensor(parameters['W'+str(l+1)])
        params['b'+str(l+1)] = tf.convert_to_tensor(parameters['b'+str(l+1)])

    x = tf.placeholder("float", [X.shape[0], X.shape[1]])

    ZL = forward_propagation(x, params)
    p = tf.argmax(ZL)

    sess = tf.Session()
    prediction = sess.run(p, feed_dict = {x: X})
    sess.close()

    return prediction
