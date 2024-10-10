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

def forward_propagation(inputs, weights, biases):  # forward propagation
    activations = [inputs]
    for W, b in zip(weights, biases):
        z = [
            sum([activations[-1][i] * W[j][i] for i in range(len(activations[-1]))]) + b[j]
            for j in range(len(b))
        ]
        activations.append([relu(z_i) for z_i in z])
    return activations

def backward_propagation(activations, y_true, weights, biases, learning_rate):  # backward propagation
    output_layer = activations[-1]
    errors = [
        (output_layer[i] - y_true[i]) * sigmoid_derivative(output_layer[i])
        for i in range(len(y_true))
    ]
    deltas = [errors]
    # Backpropagating the hidden layer error
    for l in range(len(weights)-1, 0, -1):
        hidden_errors = [
            sum([deltas[0][k] * weights[l][k][j] for k in range(len(deltas[0]))]) * relu_derivative(activations[l][j])
            for j in range(len(activations[l]))
        ]
        deltas.insert(0, hidden_errors)
    # Update the weights and biases
    for l in range(len(weights)):
        for i in range(len(weights[l])):
            for j in range(len(weights[l][i])):
                weights[l][i][j] -= learning_rate * deltas[l][i] * activations[l][j]
            biases[l][i] -= learning_rate * deltas[l][i]

    return weights, biases

def train(X, y, layer_sizes, epochs, learning_rate):
    weights, biases = initialize_weights(layer_sizes)

    for epoch in range(epochs):
        total_loss = 0
        for i in range(len(X)):
            activations = forward_propagation(X[i], weights, biases)
            total_loss += cross_entropy_loss(y[i], activations[-1])
            weights, biases = backward_propagation(activations, y[i], weights, biases, learning_rate)
        
        m = (epoch + (epochs // 20) - 1) // (epochs // 20)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(X)}")
        print(f"[{'+'*m}{' '*(20-m)}]")
        print("\033[3A")
    print("\n\nComplete")
    return weights, biases


# DataSet for XOR
X = [[0, 0], [0, 1], [1, 0], [1, 1]]  # Input
y = [[0], [1], [1], [0]]  # Output

epochs = 500  # Number of epochs
learning_rate = 0.01  # Learning rate
layer_sizes = [2, 8, 16, 8, 1]  # 2 input -> 8 hidden -> 16 hidden -> 8 hidden -> 1 output

weights, biases = train(X, y, layer_sizes, epochs, learning_rate)

for i in range(len(X)):  # Prediction
    activations = forward_propagation(X[i], weights, biases)
    output = activations[-1]
    binary_output = [1 if o >= 0.5 else 0 for o in output]
    print(f"Inputs: {X[i]}, Output: {y[i]} Predict: {binary_output}, probability: {round(output[0], 3)}")
