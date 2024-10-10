import random

class NeuralNetwork:
    def __init__(self, layer_sizes):
        self.layer_sizes = layer_sizes
        self.napier_number = self.napiers_logarithm(1000000000)  # e
        self.weights, self.biases = self.initialize_weights()
        self.activations = []

    def napiers_logarithm(self, x):  # e = (1 + 1/x)^x
        return (1 + 1 / x) ** x

    def sigmoid(self, x):  # f(x) = 1 / (1 + e^-x)
        return 1 / (1 + self.napier_number ** -x)

    def sigmoid_derivative(self, x):  # f'(x) = f(x) * (1 - f(x))
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    def relu(self, x):  # f(x) = max(0, x)
        return max(0, x)

    def relu_derivative(self, x):  # f'(x) = 1 if x > 0 else 0
        return 1 if x > 0 else 0

    def ln(self, x, n_terms=100):
        if x <= 0:
            raise ValueError("x must be positive")
        elif x == 1:
            return 0  # ln(1) = 0
        
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
        self.activations = [inputs]
        for W, b in zip(self.weights, self.biases):
            z = [
                sum([self.activations[-1][i] * W[j][i] for i in range(len(self.activations[-1]))]) + b[j]
                for j in range(len(b))
            ]
            self.activations.append([self.relu(z_i) for z_i in z])

    def backward_propagation(self, y_true, learning_rate):  # backward propagation
        output_layer = self.activations[-1]
        errors = [
            (output_layer[i] - y_true[i]) * self.sigmoid_derivative(output_layer[i])
            for i in range(len(y_true))
        ]
        deltas = [errors]
        # Backpropagating the hidden layer error
        for l in range(len(self.weights)-1, 0, -1):
            hidden_errors = [
                sum([deltas[0][k] * self.weights[l][k][j] for k in range(len(deltas[0]))]) * self.relu_derivative(self.activations[l][j])
                for j in range(len(self.activations[l]))
            ]
            deltas.insert(0, hidden_errors)
        # Update the weights and biases
        for l in range(len(self.weights)):
            for i in range(len(self.weights[l])):
                for j in range(len(self.weights[l][i])):
                    self.weights[l][i][j] -= learning_rate * deltas[l][i] * self.activations[l][j]
                self.biases[l][i] -= learning_rate * deltas[l][i]

    def train(self, X, y, epochs, learning_rate):
        for epoch in range(epochs):
            total_loss = 0
            for i in range(len(X)):
                self.forward_propagation(X[i])
                total_loss += self.cross_entropy_loss(y[i], self.activations[-1])
                self.backward_propagation(y[i], learning_rate)
            
            bar_count = (epoch + (epochs // 20) - 1) // (epochs // 20)
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(X)}")
            print(f"[{'+'*bar_count}{' '*(20-bar_count)}]")  # Progress bar
            print("\033[3A")  # Move the cursor up 3 lines
        print("\n\nComplete")

layer_sizes = [2, 8, 16, 8, 1]  # 2 input -> 8 hidden -> 16 hidden -> 8 hidden -> 1 output

# DataSet for XOR
X = [[0, 0], [0, 1], [1, 0], [1, 1]]  # Input
y = [[0], [1], [1], [0]]  # Output

epochs = 500  # Number of epochs
learning_rate = 0.01  # Learning rate

nn = NeuralNetwork(layer_sizes)
nn.train(X, y, epochs, learning_rate)

for i in range(len(X)):  # Prediction
    nn.forward_propagation(X[i])
    output = nn.activations[-1]
    binary_output = [1 if o >= 0.5 else 0 for o in output]
    print(f"Inputs: {X[i]}, Output: {y[i]}, Predict: {binary_output}, probability: {round(output[0], 5)}")
