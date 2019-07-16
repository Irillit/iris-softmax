from datapackage import Package
import numpy as np
import random

def init_weights():
    W1 = np.random.randn(4, 4) * 0.01
    b1 = np.zeros([4, 1])

    W2 = np.random.randn(3, 4) * 0.01
    b2 = np.zeros([3, 1])
    
    return W1, b1, W2, b2

def softmax(Z):
    Z_exp = np.exp(Z)
    denominator = np.sum(Z_exp)
    result = Z_exp / denominator
    return result

def relu(Z):
    return Z * (Z > 0)

def forward_propagation(X, W1, b1, W2, b2):
    Z1 = np.dot(W1, X) + b1
    A1 = relu(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = softmax(Z2)

    return A1, A2

def backward_propagation(X, A1, A2, Y,  W1, b1, W2, b2, learning_rate):
    m = A2.shape[1]
    dZ2 = A2 - train_y 

    dW2 = np.dot(dZ2, A1.T)/ m
    db2 = np.sum(dZ2, axis = 1, keepdims = True) / m
    dZ1 = np.dot(W2.T, dZ2) * (1 - np.power(A1, 2))
    dW1 = np.dot(dZ1, X.T)/ m
    db1 = np.sum(dZ1, axis = 1, keepdims = True) / m

    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2

    return W1, b1, W2, b2

def cost_function(Y, A2):
    m = Y.shape[1]
    logprobs = np.multiply(np.log(A2), Y) + np.multiply(np.log(1 - A2) ,(1 - Y))
    cost = - (1/ m) * np.sum(logprobs, axis = 1, keepdims = True)

    #cost = float(np.squeeze(cost))

    return cost

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

    return train_x.T, train_y.T, test_x.T, test_y.T

train_x, train_y, test_x, test_y = get_training_and_test_data()

print("X: " + str(train_x.shape) + " Y: " + str(train_y.shape))
W1, b1, W2, b2 = init_weights()

print("W1: "  + str(W1.shape) + " b1 " + str(b1.shape) + " W2: "  + str(W2.shape) + " b2 " + str(b2.shape)) 
for i in range(100):
    A1, A2 = forward_propagation(train_x, W1, b1, W2, b2)
    cost = cost_function(train_y, A2)
    print("#" +str(i) + str(cost))
    W1, b1, W2, b2 = backward_propagation(train_x, A1, A2, train_y,  W1, b1, W2, b2, 1.1)
    
    


