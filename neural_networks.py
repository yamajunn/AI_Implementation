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