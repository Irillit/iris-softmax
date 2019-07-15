from datapackage import Package
import numpy as np
import random

def init_weights():
    W1 = np.random.randn(4, 4) * 0.01
    b1 = np.zeros([4, 105])

    W2 = np.random.randn(3, 4) * 0.01
    b2 = np.zeros([3, 105])
    
    return W1, b1, W2, b2

def softmax(Z):
    Z_exp = np.exp(Z)
    denominator = np.sum(Z_exp)
    result = Z_exp / denominator
    return result

def relu(Z):
    return Z * (Z > 0)

def forward_propagation(X, W1, b1, W2, b2):
    Z1 = np.dot(W1, X.T) + b1
    A1 = relu(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = softmax(Z2)

    return A2

def back_propagation(A2, Y,  W1, b1, W2, b2):
    m = A2.shape[1]
    dZ = train_y - A2

    dW2 = np.dot(dZ, A2)/ m
    db2 = np.sum(dZ, axis = 1, keepdims = True) / m
    dA_prev = np.dot(W2.T, dZ)

def get_dataset():
    package = Package('https://datahub.io/machine-learning/iris/datapackage.json')
    irises = None
    for resource in package.resources:
        if resource.descriptor['datahub']['type'] == 'derived/csv':
            irises = resource.read()
    return irises

def get_training_and_test_data():
    iris_class = {
        'Iris-setosa': [1, 0, 0],
        'Iris-versicolor': [0, 1, 0],
        'Iris-virginica': [0, 0, 1]
        }
    X = []
    Y = []

    raw_irises = get_dataset()
    random.shuffle(raw_irises)

    for iris in raw_irises:
        x = iris[0:4]
        X.append(x)
        y = iris_class[iris[4]]
        Y.append(y)

    train_x = X[0:105]
    train_y = Y[0:105]

    test_x = X[105:150]
    test_y = Y[105:150]

    train_x = np.array(train_x, dtype=float)
    train_y = np.array(train_y, dtype=float)
    test_x = np.array(test_x, dtype=float)
    test_y = np.array(test_y, dtype=float)

    return train_x, train_y.T, test_x, test_y.T

train_x, train_y, test_x, test_y = get_training_and_test_data()

W1, b1, W2, b2 = init_weights()

A2 = forward_propagation(train_x, W1, b1, W2, b2)

print (dZ)

