import random

def napiers_logarithm(x):  # e = (1 + 1/x)^x
    return (1 + 1 / x) ** x
napier_number = napiers_logarithm(1000000000)  # e

def sigmoid(x):  # f(x) = 1 / (1 + e^-x)
    return 1 / (1 + napier_number ** -x)

def sigmoid_derivative(x):  # f'(x) = f(x) * (1 - f(x))
    return sigmoid(x) * (1 - sigmoid(x))

def relu(x):  # f(x) = max(0, x)
    return max(0, x)

def relu_derivative(x):  # f'(x) = 1 if x > 0 else 0
    return 1 if x > 0 else 0

def ln(x, n_terms=10000):  # ln(x) = x - x^2/2 + x^3/3 - x^4/4 + ...
    if x <= 0: raise ValueError("Input must be greater than 0.")
    x -= 1
    return sum([((-1)**(n+1))*(x**n)/n for n in range(1, n_terms + 1)])

def cross_entropy_loss(y_true, y_pred):  # -sum(y_true[i] * ln(y_pred[i] + 1e-9))
    if len(y_true) != len(y_pred): raise ValueError("Input lists must have the same length.")
    return -sum([y_true[i] * ln(y_pred[i] + 1e-9) for i in range(len(y_true))])


def initialize_weights(layer_sizes):  # initialize weights and biases
    weights, biases = [], []
    for i in range(len(layer_sizes) - 1):
        W = [[random.uniform(-1, 1) for _ in range(layer_sizes[i])] for _ in range(layer_sizes[i+1])]  # [ {random_num *layer_sizes[i]} *layer_sizes[i+1] ]
        b = [random.uniform(-1, 1) for _ in range(layer_sizes[i+1])]  # [ random_num *layer_sizes[i+1] ]
        weights.append(W)
        biases.append(b)
    return weights, biases

def forward_propagation(inputs, weights, biases):
    activations = [inputs]
    for W, b in zip(weights, biases):
        z = [sum([activations[-1][i] * W[j][i] for i in range(len(activations[-1]))]) + b[j] for j in range(len(b))]
        activations.append([relu(z_i) for z_i in z])
    return activations

'''     2   3   3   1  <- layer_sizes
        #   #   #
2_input     #   #   #  1_output
        #   #   #
'''