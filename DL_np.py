import numpy as np
import matplotlib.pyplot as plt


def sigmoid(Z):

    A = 1/(1+np.exp(-Z))
    cache = Z

    return A, cache

def relu(Z):

    A = np.maximum(0,Z)

    assert(A.shape == Z.shape)

    cache = Z
    return A, cache


def initialize_parameters_he(layers_dims):

    np.random.seed(4)
    parameters = {}
    L = len(layers_dims) - 1 # integer representing the number of layers

    for l in range(1, L + 1):

        parameters['W' + str(l)] = np.random.randn(layers_dims[l], layers_dims[l-1]) * (np.sqrt(2. / layers_dims[l-1]))
        parameters['b' + str(l)] = np.zeros((layers_dims[l], 1))

    return parameters



def initialize_parameters(layers_dims):

    np.random.seed(4)
    parameters = {}
    L = len(layers_dims) - 1 # integer representing the number of layers

    for l in range(1, L + 1):

        parameters['W' + str(l)] = np.random.randn(layers_dims[l], layers_dims[l-1]) * 0.01
        parameters['b' + str(l)] = np.zeros((layers_dims[l], 1))

    return parameters




def activation_forward(A_prev, W, b, activation):

    if activation == "sigmoid":

        Z = np.dot(W, A_prev) + b

        assert(Z.shape == (W.shape[0], A_prev.shape[1]))

        linear_cache = (A_prev, W, b)

        A, activation_cache = sigmoid(Z)


    elif activation == "relu":

        Z = np.dot(W, A_prev) + b

        assert(Z.shape == (W.shape[0], A_prev.shape[1]))

        linear_cache = (A_prev, W, b)

        A, activation_cache = relu(Z)

    assert (A.shape == (W.shape[0], A_prev.shape[1]))
    cache = (linear_cache, activation_cache)

    return A, cache



def activation_forward_with_dropout(A_prev, W, b, activation, keep_prob):

    np.random.seed(1)

    Z = np.dot(W, A_prev) + b

    assert(Z.shape == (W.shape[0], A_prev.shape[1]))

    linear_cache = (A_prev, W, b)

    if activation == "sigmoid":

        A, activation_cache = sigmoid(Z)

        cache = (linear_cache, activation_cache)

    elif activation == "relu":

        A, activation_cache = relu(Z)

        D = np.random.rand(A.shape[0], A.shape[1])
        D = D < keep_prob
        A = np.multiply(A, D)
        A /= keep_prob

        cache = (linear_cache, activation_cache, D)

    assert (A.shape == (W.shape[0], A_prev.shape[1]))

    return A, cache



def model_forward(X, parameters):

    caches = []
    A = X
    L = len(parameters) // 2

    for l in range(1, L):
        A_prev = A

        A, cache = activation_forward(A_prev,
                                      parameters['W' + str(l)],
                                      parameters['b' + str(l)],
                                      activation='relu')
        caches.append(cache)

    AL, cache = activation_forward(A,
                                   parameters['W' + str(L)],
                                   parameters['b' + str(L)],
                                   activation='sigmoid')
    caches.append(cache)

    assert(AL.shape == (1, X.shape[1]))

    return AL, caches


def model_forward_with_dropout(X, parameters, keep_prob=0.5):

    caches = []
    A = X
    L = len(parameters) // 2

    for l in range(1, L):
        A_prev = A

        A, cache = activation_forward_with_dropout(A_prev,
                                      parameters['W' + str(l)],
                                      parameters['b' + str(l)],
                                      activation='relu',
                                      keep_prob = keep_prob)
        caches.append(cache)

    AL, cache = activation_forward_with_dropout(A,
                                   parameters['W' + str(L)],
                                   parameters['b' + str(L)],
                                   activation='sigmoid',
                                   keep_prob = keep_prob)
    caches.append(cache)

    assert(AL.shape == (1, X.shape[1]))

    return AL, caches


def compute_cost(AL, Y):

    m = Y.shape[1]

    #cost = (-1 / m) * (np.dot(Y, np.log(AL).T) + np.dot(1 - Y, np.log(1 - AL).T))

    #cost = np.squeeze(cost)

    # deal with 0 derivative
    logprobs = np.multiply(-np.log(AL), Y) + np.multiply(-np.log(1-AL), 1 - Y)
    cost = 1./m * np.nansum(logprobs)

    assert(cost.shape == ())

    return cost


def compute_cost_with_regularization(AL, Y, parameters, lambd):

    m = Y.shape[1]
    cross_entropy_cost = compute_cost(AL, Y)
    L2_regularization_cost = 0

    for l in range(len(parameters)/2):
        L2_regularization_cost += np.sum(np.square(parameters["W" + str(l+1)]))

    L2_regularization_cost = L2_regularization_cost * (1. / m) * (lambd / 2)

    cost = cross_entropy_cost + L2_regularization_cost

    return cost


def relu_backward(dA, cache):

    Z = cache

    dZ = np.array(dA, copy=True)
    dZ = np.multiply(dZ, np.int64(Z >= 0))

    assert (dZ.shape == Z.shape)

    return dZ

def sigmoid_backward(dA, cache):

    Z = cache

    s = 1/(1+np.exp(-Z))
    dZ = dA * s * (1-s)

    assert (dZ.shape == Z.shape)

    return dZ

def activation_backward(dA, cache, activation):

    linear_cache, activation_cache = cache

    if activation == "relu":

        dZ = relu_backward(dA, activation_cache)

    elif activation == "sigmoid":

        dZ = sigmoid_backward(dA, activation_cache)


    A_prev, W, b = linear_cache
    m = A_prev.shape[1]

    dW = (1. / m) * np.dot(dZ, A_prev.T)
    db = (1. / m) * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)

    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)

    return dA_prev, dW, db


def activation_backward_with_regularization(dA, cache, activation, lambd):

    linear_cache, activation_cache = cache

    if activation == "relu":

        dZ = relu_backward(dA, activation_cache)


    elif activation == "sigmoid":

        dZ = sigmoid_backward(dA, activation_cache)

    A_prev, W, b = linear_cache
    m = A_prev.shape[1]

    dW = (1. / m) * (np.dot(dZ, A_prev.T) + lambd * W)
    db = (1. / m) * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)

    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)

    return dA_prev, dW, db



def activation_backward_with_dropout(dA, cache, activation, keep_prob):

    linear_cache, activation_cache, D = cache

    A_prev, W, b = linear_cache

    if activation == "relu":

        dA = np.multiply(dA, D)
        dA /= keep_prob

        dZ = relu_backward(dA, activation_cache)

    elif activation == "sigmoid":

        dZ = sigmoid_backward(dA, activation_cache)

    m = A_prev.shape[1]

    dW = (1. / m) * np.dot(dZ, A_prev.T)
    db = (1. / m) * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)

    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)

    return dA_prev, dW, db

def nan_divide(a, b):

    with np.errstate(divide='ignore', invalid='ignore'):
        c = np.true_divide(a,b)
        c[c == np.inf] = 0
        c = np.nan_to_num(c)
    return c


def model_backward(AL, Y, caches):

    grads = {}
    L = len(caches)
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)

    dAL = - (nan_divide(Y, AL) - nan_divide(1 - Y, 1 - AL))

    current_cache = caches[-1]
    grads["dA" + str(L)], grads["dW" + str(L)], grads["db" + str(L)] = activation_backward(dAL, current_cache, activation="sigmoid")

    for l in reversed(range(L-1)):

        current_cache = caches[l]

        dA_prev_temp, dW_temp, db_temp = activation_backward(grads["dA" + str(l + 2)], current_cache, activation="relu")
        grads["dA" + str(l + 1)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp

    return grads

def model_backward_with_regularization(AL, Y, caches, lambd):

    grads = {}
    L = len(caches)
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)

    dAL = - (nan_divide(Y, AL) - nan_divide(1 - Y, 1 - AL))

    current_cache = caches[-1]
    grads["dA" + str(L)], grads["dW" + str(L)], grads["db" + str(L)] = activation_backward_with_regularization(dAL, current_cache, activation="sigmoid", lambd = lambd)

    for l in reversed(range(L-1)):

        current_cache = caches[l]

        dA_prev_temp, dW_temp, db_temp = activation_backward_with_regularization(grads["dA" + str(l + 2)], current_cache, activation="relu", lambd = lambd)
        grads["dA" + str(l + 1)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp

    return grads


def model_backward_with_dropout(AL, Y, caches, keep_prob):

    grads = {}
    L = len(caches)
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)

    dAL = - (nan_divide(Y, AL) - nan_divide(1 - Y, 1 - AL))

    current_cache = caches[-1]
    grads["dA" + str(L)], grads["dW" + str(L)], grads["db" + str(L)] = activation_backward(dAL, current_cache, activation="sigmoid")

    for l in reversed(range(L-1)):

        current_cache = caches[l]

        dA_prev_temp, dW_temp, db_temp = activation_backward_with_dropout(grads["dA" + str(l + 2)], current_cache, activation="relu", keep_prob = keep_prob)
        grads["dA" + str(l + 1)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp

    return grads



def update_parameters(parameters, grads, learning_rate):

    L = len(parameters) // 2

    for l in range(L):
        parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - learning_rate * grads["dW" + str(l + 1)]
        parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * grads["db" + str(l + 1)]

    return parameters

def predict(X, y, parameters):

    m = X.shape[1]
    n = len(parameters) // 2
    p = np.zeros((1,m))

    probs, caches = model_forward(X, parameters)

    for i in range(0, probs.shape[1]):
        if probs[0,i] > 0.5:
            p[0,i] = 1
        else:
            p[0,i] = 0

    accuracy = np.sum((p == y)/m)

    print("Accuracy: "  + str(accuracy))

    return accuracy



def model_stopping(X, Y, valid_x, valid_y, layers_dims, learning_rate = 0.0075, num_iterations = 10000, print_cost=False, print_size=100):

    costs = []

    accuracy_prev = 0

    parameters = initialize_parameters(layers_dims)

    for i in range(0, num_iterations):

        AL, caches = model_forward(X, parameters)

        cost = compute_cost(AL, Y)

        grads = model_backward(AL, Y, caches)

        cache = parameters

        parameters = update_parameters(parameters, grads, learning_rate=learning_rate)

        if print_cost and i % print_size == 0:
            print ("Cost after iteration %i: %f" %(i, cost))

        if print_cost and i % 100 == 0:
            accuracy  = predict(valid_x, valid_y, parameters)
            costs.append(cost)

        if accuracy < accuracy_prev:
            parameters = cache
            print ("Early stopping at iteration {} to prevent overfitting".format(i))
            break
        else:
            accuracy_prev = accuracy

    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()

    return parameters



def deep_model(X, Y, layers_dims, learning_rate = 0.0075, num_iterations = 10000, print_cost=False, print_size=100, lambd=0, keep_prob=1, continue_train = False, initial_parameters = 0):

    Y = Y.reshape(1, X.shape[1])
    
    costs = []

    accuracy_prev = 0

    if continue_train == False:
        parameters = initialize_parameters(layers_dims)
    else:
        parameters = initial_parameters

    for i in range(0, num_iterations):

        if keep_prob == 1:
            AL, caches = model_forward(X, parameters)
        else:
            AL, caches = model_forward_with_dropout(X, parameters, keep_prob)

        if lambd == 0:
            cost = compute_cost(AL, Y)
        else:
            cost = compute_cost_with_regularization(AL, Y, parameters, lambd)

        # one or the other
        assert(lambd==0 or keep_prob==1)

        if lambd == 0 and keep_prob == 1:
            grads = model_backward(AL, Y, caches)
        elif lambd != 0:
            grads = model_backward_with_regularization(AL, Y, caches, lambd)
        elif keep_prob <1:
            grads = model_backward_with_dropout(AL, Y, caches, keep_prob)

        cache = parameters

        parameters = update_parameters(parameters, grads, learning_rate=learning_rate)

        if print_cost and i % print_size == 0:
            print ("Cost after iteration %i: %f" %(i, cost))

        if print_cost and i % 100 == 0:
            costs.append(cost)

    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()

    np.save("param_" + str(len(layers_dims)-1) + "layer_" + str(i), parameters)

    return parameters


# Prediction for binary outcomes
def predict_dec(parameters, X):

    # Predict using forward propagation and a classification threshold of 0.5
    a3, cache = model_forward(X, parameters)

    predictions = (a3>0.5)
    return predictions
