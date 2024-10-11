import random
import time
import copy

class NeuralNetwork:
    def __init__(self, hidden_size=[8, 16, 8], epochs=1000, learning_rate=0.01):
        self.layer_sizes = hidden_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.napier_number = self.napiers_logarithm(1000000000)  # e
        self.weights, self.biases = [], []
        self.activations = []
        self.results = []
        self.max_val, self.min_val = 0, 0

    def napiers_logarithm(self, x):  # e = (1 + 1/x)^x
        return (1 + 1 / x) ** x

    def sigmoid(self, x):
        if x > 0:
            return 1 / (1 + self.napier_number ** -x)
        else:
            # Avoid overflow
            exp_neg_x = self.napier_number ** x
            return exp_neg_x / (1 + exp_neg_x)

    def sigmoid_derivative(self, x):  # f'(x) = f(x) * (1 - f(x))
        sig = self.sigmoid(x)
        return sig * (1 - sig)

    def leaky_relu(self, x, alpha=0.01):  # f(x) = x if x > 0 else alpha * x
        return x if x > 0 else alpha * x

    def leaky_relu_derivative(self, x, alpha=0.01):  # f'(x) = 1 if x > 0 else alpha
        return 1 if x > 0 else alpha

    def ln(self, x, n_terms=100):
        if x <= 0: raise ValueError("x must be positive")
        elif x == 1: return 0  # ln(1) = 0
        # decompose into ln(x / 2^k)
        result, factor = 0, 0
        while x > 2:
            x /= 2
            factor += 1
        # teilor expansion
        z = x - 1
        for n in range(1, n_terms + 1):
            term = ((-1) ** (n + 1)) * (z ** n) / n
            result += term
        return result + factor * 0.69314718056  # ln(2) = 0.69314718056


    def cross_entropy_loss(self, y_true, y_pred):  # -sum(y_true[i] * ln(y_pred[i] + 1e-9))
        if len(y_true) != len(y_pred): raise ValueError("Input lists must have the same length.")
        return -sum([y_true[i] * self.ln(y_pred[i] + 1e-9) for i in range(len(y_true))])

    def initialize_weights(self):  # initialize weights and biases
        weights, biases = [], []
        for i in range(len(self.layer_sizes) - 1):
            W = [[random.uniform(-1, 1) for _ in range(self.layer_sizes[i])] for _ in range(self.layer_sizes[i+1])]  # [ {random_num *layer_sizes[i]} *layer_sizes[i+1] ]
            b = [random.uniform(-1, 1) for _ in range(self.layer_sizes[i+1])]  # [ random_num *layer_sizes[i+1] ]
            weights.append(W)
            biases.append(b)
        return weights, biases

    def forward_propagation(self, inputs):  # forward propagation
        # Hidden layers
        self.activations = [inputs]
        for W, b in zip(self.weights, self.biases[:-1]):
            z = [
                sum([self.activations[-1][i] * W[j][i] for i in range(len(self.activations[-1]))]) + b[j]
                for j in range(len(b))
            ]
            self.activations.append([self.leaky_relu(z_i) for z_i in z])
        
        # Output layer
        W, b = self.weights[-1], self.biases[-1]
        z = [
            sum([self.activations[-1][i] * W[j][i] for i in range(len(self.activations[-1]))]) + b[j]
            for j in range(len(b))
        ]
        self.activations.append([self.sigmoid(z_i) for z_i in z])  # Sigmoid activation function

    def backward_propagation(self, y_true):  # backward propagation
        output_layer = self.activations[-1]
        errors = [
            (output_layer[i] - y_true[i]) * self.sigmoid_derivative(output_layer[i])
            for i in range(len(y_true))
        ]
        deltas = [errors]
        # Backpropagating the hidden layer error
        for l in range(len(self.weights)-1, 0, -1):
            hidden_errors = [
                sum([deltas[0][k] * self.weights[l][k][j] for k in range(len(deltas[0]))]) * self.leaky_relu_derivative(self.activations[l][j])
                for j in range(len(self.activations[l]))
            ]
            deltas.insert(0, hidden_errors)
        # Update the weights and biases
        for l in range(len(self.weights)):
            for i in range(len(self.weights[l])):
                for j in range(len(self.weights[l][i])):
                    self.weights[l][i][j] -= self.learning_rate * deltas[l][i] * self.activations[l][j]
                self.biases[l][i] -= self.learning_rate * deltas[l][i]

    def train(self, X, y):
        self.layer_sizes.insert(0, len(X[0]))  # Add the input layer size
        self.layer_sizes.append(len(y[0]))  # Add the output layer size
        self.weights, self.biases = self.initialize_weights()
        self.y_true = y
        X = self.normalize(X)
        start = time.time()
        for epoch in range(self.epochs):
            total_loss = 0
            for i in range(len(X)):
                self.forward_propagation(X[i])  # Forward propagation
                total_loss += self.cross_entropy_loss(y[i], self.activations[-1])  # Calculate the total loss
                self.backward_propagation(y[i])  # Backward propagation
            
            bar_count = (epoch + (self.epochs // 20) - 1) // (self.epochs // 20)
            print(f"Epoch {epoch+1}/{self.epochs}, Loss: {total_loss/len(X)}")
            print(f"[{'+'*bar_count}{' '*(20-bar_count)}]")  # Progress bar
            print("\033[3A")  # Move the cursor up 3 lines
        print(f"\n\nTime: {time.time() - start:.2f}s")
    
    def predict(self, X):
        self.results = []
        X = self.normalize_recursive(X)
        for i in range(len(X)):
            self.forward_propagation(X[i])
            self.results.append(self.activations[-1])
        return self.results
    
    def accuracy(self, y_true):
        accuracy = 0
        for i in range(len(y_true)):
            binary_output = [1 if self.results[i][j] >= 0.5 else 0 for j in range(len(y_true[i]))]
            accuracy += sum([1 if y_true[i][j] == binary_output[j] else 0 for j in range(len(y_true[i]))]) / len(y_true[i])
        return accuracy / len(y_true) * 100
    
    def normalize_recursive(self, lst):
        for i in range(len(lst)):
            if isinstance(lst[i], list):
                self.normalize_recursive(lst[i])
            else:
                lst[i] = 2 * (lst[i] - self.min_val) / (self.max_val - self.min_val) - 1
        return lst
    def normalize(self, X):
        def flatten(lst):
            flat_list = []
            for item in lst:
                if isinstance(item, list):
                    flat_list.extend(flatten(item))
                else:
                    flat_list.append(item)
            return flat_list

        X_sum = copy.deepcopy(X)
        flat_X = flatten(X_sum)
        self.max_val = max(flat_X)
        self.min_val = min(flat_X)
        X = self.normalize_recursive(X)
        return X

# DataSet for XOR
X = [[0, 0], [0, 1], [1, 0], [1, 1]]  # Input
y = [[0], [1], [1], [0]]  # Output
x_test = [[0, 0], [0, 1], [1, 0], [1, 1]]
y_test = [[0], [1], [1], [0]]

# 2 input -> 8 hidden -> 16 hidden -> 8 hidden -> 1 output
nn = NeuralNetwork(hidden_size=[4, 8, 16, 32, 64, 32, 16, 8, 4], epochs=1000, learning_rate = 0.01)  # Create a neural network
nn.train(X, y)  # Train the neural network

accuracy = 0
output = nn.predict(copy.deepcopy(x_test))  # Predict the output
output_copy = copy.deepcopy(output)
for i in range(len(x_test)):
    output_copy[i] = [1 if output[i][j] >= 0.5 else 0 for j in range(len(output[i]))]
    print(f"Input: {x_test[i]}, Output: {output_copy[i]}, Probability: {output[i]}")
print(f"Accuracy: {nn.accuracy(y_test):.2f}%")