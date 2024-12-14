import random
import time

def napiers_logarithm(x):
    """
    ネイピア数を求める関数

    Args:
        x (float): 入力
    
    Returns:
        float: 出力
    """
    return (1 + 1 / x) ** x
napier_number = napiers_logarithm(100000000)  # e

def ln(x, max_iter=20, tol=1e-12):
    """
    自然対数を求める関数

    Args:
        x (float): 入力
        max_iter (int): 最大反復回数
        tol (float): 許容誤差
    
    Returns:
        float: 出力
    """
    if x <= 0: raise ValueError("x must be positive")
    k = 0
    while x > 2:
        x /= 2
        k += 1
    while x < 0.5:
        x *= 2
        k -= 1
    y = x - 1  # ln(1) = 0 付近の値から開始
    for _ in range(max_iter):
        prev_y = y
        y -= (2.718281828459045**y - x) / (2.718281828459045**y)  # f(y) / f'(y)
        if abs(y - prev_y) < tol:
            break
    return y + k * 0.6931471805599453  # ln(2) ≈ 0.693147

def sqrt(x):
    """
    平方根を求める関数

    Args:
        x (float): 入力
    
    Returns:
        float: 出力
    """
    tolerance = 1e-10  # 許容誤差
    estimate = x / 2.0  # 初期推定値
    while True:
        new_estimate = (estimate + x / estimate) / 2  # ニュートン法による更新
        if abs(new_estimate - estimate) < tolerance:  # 収束したら終了
            return new_estimate
        estimate = new_estimate  # 推定値を更新

def sigmoid(x, derivative=False):
    """
    Sigmoid 関数およびその微分

    Args:
        x (float): 入力
        derivative (bool): 微分を計算するかどうか
    
    Returns:
        float: 出力
    """
    if derivative:
        return sigmoid_derivative(x)
    return 1 / (1 + napier_number ** -x)

def sigmoid_derivative(x):
    """
    Sigmoid 関数の微分

    Args:
        x (float): 入力
    
    Returns:
        float: 出力
    """
    return sigmoid(x) * (1 - sigmoid(x))

def relu(x, derivative=False):
    """
    ReLU 関数およびその微分

    Args:
        x (float): 入力
        derivative (bool): 微分を計算するかどうか
    
    Returns:
        float: 出力
    """
    if derivative:
        return relu_derivative(x)
    return max(0, x)

def relu_derivative(x):
    """
    ReLU 関数の微分

    Args:
        x (float): 入力
    
    Returns:
        float: 出力
    """
    return 1 if x > 0 else 0

def leaky_relu(x, alpha=0.01, derivative=False):
    """
    Leaky ReLU 関数およびその微分

    Args:
        x (float): 入力
        alpha (float): ハイパーパラメータ
        derivative (bool): 微分を計算するかどうか
    
    Returns:
        float: 出力
    """
    if derivative:
        return leaky_relu_derivative(x, alpha)
    return x if x > 0 else alpha * x

def leaky_relu_derivative(x, alpha=0.01):
    """
    Leaky ReLU 関数の微分

    Args:
        x (float): 入力
        alpha (float): ハイパーパラメータ
    
    Returns:
        float: 出力
    """
    return 1 if x > 0 else alpha

def identity(x, derivative=False):
    """
    恒等関数およびその微分

    Args:
        x (float): 入力
        derivative (bool): 微分を計算するかどうか
    
    Returns:
        float: 出力
    """
    if derivative:
        return identity_derivative(x)
    return x

def identity_derivative(x):
    """
    恒等関数の微分

    Args:
        x (float): 入力(未使用)
    
    Returns:
        int: 出力
    """
    return 1

def cross_entropy_loss(y_true, y_pred):
    """
    交差エントロピー損失関数

    Args:
        y_true (list): 正解ラベル
        y_pred (list): 予測ラベル
    
    Returns:
        float: 出力
    """
    if len(y_true) != len(y_pred): raise ValueError("Input lists must have the same length.")
    return -sum([t * ln(p + 1e-9) for t, p in zip(y_true, y_pred)])

def mean_squared_error(y_true, y_pred):
    """
    平均二乗誤差を求める関数

    Args:
        y_true (list): 正解ラベル
        y_pred (list): 予測ラベル
    
    Returns:
        float: 出力
    """
    if len(y_true) != len(y_pred): raise ValueError("Input lists must have the same length.")
    return sum([(t - p) ** 2 for t, p in zip(y_true, y_pred)]) / len(y_true)

def mean_absolute_error(y_true, y_pred):
    """
    平均絶対誤差を求める関数

    Args:
        y_true (list): 正解ラベル
        y_pred (list): 予測ラベル
    
    Returns:
        float: 出力
    """
    if len(y_true) != len(y_pred): raise ValueError("Input lists must have the same length.")
    return sum([abs(t - p) for t, p in zip(y_true, y_pred)]) / len(y_true)

def binary_cross_entropy_loss(y_true, y_pred):
    """
    バイナリ交差エントロピー損失関数

    Args:
        y_true (list): 正解ラベル
        y_pred (list): 予測ラベル
    
    Returns:
        float: 出力
    """
    if len(y_true) != len(y_pred): raise ValueError("Input lists must have the same length.")
    epsilon = 1e-9  # 0で割るのを防ぐための小さな値
    return -sum([t * ln(p + epsilon) + (1 - t) * ln(1 - p + epsilon) for t, p in zip(y_true, y_pred)]) / len(y_true)

def categorical_cross_entropy_loss(y_true, y_pred):
    """
    カテゴリカル交差エントロピー損失関数

    Args:
        y_true (list): 正解ラベル
        y_pred (list): 予測ラベル
    
    Returns:
        float: 出力
    """
    if len(y_true) != len(y_pred): raise ValueError("Input lists must have the same length.")
    epsilon = 1e-9  # 0で割るのを防ぐための小さな値
    return -sum([t * ln(p + epsilon) for t, p in zip(y_true, y_pred)]) / len(y_true)

def initialize_weights(layer_sizes):  # 重みとバイアスの初期化
    """
    重みとバイアスを初期化する関数

    Args:
        layer_sizes (list): 各層のユニット数
    
    Returns:
        tuple: 重みとバイアス
    """
    weights, biases = [], []
    for i in range(len(layer_sizes) - 1):
        limit = sqrt(6 / (layer_sizes[i] + layer_sizes[i+1]))  # 重みの初期化に使う乱数の範囲
        weights.append([[random.uniform(-limit, limit) for _ in range(layer_sizes[i])] for _ in range(layer_sizes[i+1])])  # 重みは -limit から limit の間の乱数で初期化
        biases.append([0 for _ in range(layer_sizes[i+1])])  # バイアスは0で初期化
    return weights, biases

def forward_propagation(inputs, weights, biases, hidden_activation, output_activation):  # 順伝播処理
    """
    順伝播処理を行う関数

    Args:
        inputs (list): 入力
        weights (list): 重み
        biases (list): バイアス
    
    Returns:
        list: 出力
    """
    activations = [inputs]
    for W, b in zip(weights, biases):
        z = [
            sum([activations[-1][i] * W[j][i] for i in range(len(activations[-1]))]) + b[j]
            for j in range(len(b))
        ]
        if W != weights[-1]:
            activations.append([hidden_activation(z_i, derivative=False) for z_i in z])
        else:
            activations.append([output_activation(z_i, derivative=False) for z_i in z])
    return activations

def backward_propagation(activations, y_true, weights, biases, learning_rate, hidden_activation, output_activation):  # 逆伝播処理
    """
    逆伝播処理を行う関数

    Args:
        activations (list): 出力
        y_true (list): 正解ラベル
        weights (list): 重み
        biases (list): バイアス
        learning_rate (float): 学習率
    
    Returns:
        tuple: 重みとバイアス
    """
    output_layer = activations[-1]
    errors = [
        (output_layer[i] - y_true[i]) * output_activation(output_layer[i], derivative=True)
        for i in range(len(y_true))
    ]
    deltas = [errors]
    # 隠れ層の誤差を計算
    for l in range(len(weights)-1, 0, -1):
        hidden_errors = [
            sum([deltas[0][k] * weights[l][k][j] for k in range(len(deltas[0]))]) * hidden_activation(activations[l][j], derivative=True)
            for j in range(len(activations[l]))
        ]
        deltas.insert(0, hidden_errors)
    # 重みとバイアスを更新
    for l in range(len(weights)):
        for i in range(len(weights[l])):
            for j in range(len(weights[l][i])):
                weights[l][i][j] -= learning_rate * deltas[l][i] * activations[l][j]
            biases[l][i] -= learning_rate * deltas[l][i]

    return weights, biases

def train(X, y, layer_sizes, epochs, learning_rate, hidden_activation=relu, output_activation=sigmoid, loss=cross_entropy_loss):  # 学習
    """
    ニューラルネットワークを学習する関数

    Args:
        X (list): 入力
        y (list): 正解ラベル
        layer_sizes (list): 各層のユニット数
        epochs (int): エポック数
        learning_rate (float): 学習率
    
    Returns:
        tuple: 重みとバイアス
    """
    weights, biases = initialize_weights(layer_sizes)
    start = time.time()
    for epoch in range(epochs):
        total_loss = 0
        for i in range(len(X)):
            activations = forward_propagation(X[i], weights, biases, hidden_activation, output_activation)
            total_loss += loss(y[i], activations[-1])
            weights, biases = backward_propagation(activations, y[i], weights, biases, learning_rate, hidden_activation, output_activation)
        m = epoch // (epochs // 20) + 1
        print(f"\rEpoch {epoch+1}/{epochs}, Loss: {total_loss/len(X):.10f}, [{'+'*m}{' '*(20-m)}]{' '*5}",end="")
    print(f"Training time: {time.time()-start:.2f} seconds")
    return weights, biases

def predict(X, weights, biases, hidden_activation=relu, output_activation=sigmoid):  # 予測
    """
    予測を行う関数

    Args:
        X (list): 入力
        weights (list): 重み
        biases (list): バイアス
    
    Returns:
        list: 出力
    """
    outputs = []
    for i in range(len(X)):  # Prediction
        outputs.append(forward_propagation(X[i], weights, biases, hidden_activation, output_activation)[-1])
    return outputs

def accuracy(X, y, predict, loss=cross_entropy_loss):  # 予測精度の計算
    """
    予測精度を計算する関数

    Args:
        X (list): 入力
        y (list): 正解ラベル
        predict (list): 予測値
    
    Returns:
        float: 予測精度
    """
    total_loss = 0
    for i in range(len(predict)):  # Prediction
        print(f"入力: {X[i]}, 正解: {y[i]}, 予測値: {predict[i]}")
        total_loss += loss(y[i], predict[i])
    print(f"Loss: {total_loss / len(predict):.10f}")
    return total_loss / len(predict)

def normalize(data, denomalize=False, min_val=None, max_val=None):
    """
    データを正規化する関数

    Args:
        data (list): 入力データ
        denomalize (bool): 逆正規化を行うかどうか
        min_val (float): 最小値
        max_val (float): 最大値
    
    Returns:
        tuple: 正規化されたデータ、最小値、最大値もしくは逆正規化されたデータ
    """
    if denomalize:
        return [[x * (max_val - min_val) + min_val for x in sublist] for sublist in data]
    min_val = min(min(sublist) for sublist in data)
    max_val = max(max(sublist) for sublist in data)
    nomalized_data = [[(x - min_val) / (max_val - min_val) for x in sublist] for sublist in data]
    return nomalized_data, min_val, max_val

def standardize(data, unstandardize=False, mean=None, std_dev=None):
    """
    データを標準化する関数

    Args:
        data (list): 入力データ
        unstandardize (bool): 逆標準化を行うかどうか
        mean (float): 平均
        std_dev (float): 標準偏差
    
    Returns:
        tuple: 標準化されたデータ、平均、標準偏差もしくは逆標準化されたデータ
    """
    if unstandardize:
        return [[x * std_dev + mean for x in sublist] for sublist in data]
    mean = sum(sum(sublist) for sublist in data) / (len(data) * len(data[0]))
    std_dev = (sum((x - mean) ** 2 for sublist in data for x in sublist) / (len(data) * len(data[0]))) ** 0.5
    standardized_data = [[(x - mean) / std_dev for x in sublist] for sublist in data]
    return standardized_data, mean, std_dev

def label_encoding(labels, decoding=False, label_to_index=None):
    """
    ラベルエンコーディングを行う関数

    Args:
        labels (list): カテゴリカルデータのリスト(デコードを行う際はエンコードした元文字列(現数値)のみを指定可能)
        decoding (bool): デコードを行うかどうか(デコードを行う際は label_to_index を指定)
        label_to_index (dict): ラベルとインデックスのマッピング
    
    Returns:
        tuple: エンコードされたラベル、ラベルとインデックスのマッピング
    """
    if decoding:
        return [[list(label_to_index.keys())[list(label_to_index.values()).index(label)] for label in sublist] for sublist in labels]
    str_labels = [label for label in sum(labels, []) if type(label) == str]
    label_to_index = {label: idx for idx, label in enumerate(sorted(set(str_labels)))}
    for i in range(len(labels)):
        for j in range(len(labels[i])):
            if labels[i][j] in label_to_index and type(labels[i][j]) == str:
                labels[i][j] = label_to_index[labels[i][j]]
    return labels, label_to_index

def one_hot_encoding(labels, decoding=False, label_to_index=None):
    """
    ワンホットエンコーディングを行う関数
    
    Args:
        labels (list): カテゴリカルデータのリスト(デコードを行う際はエンコードした元文字列(現数値)のみを指定可能)
        decoding (bool): デコードを行うかどうか(デコードを行う際は label_to_index を指定)
        label_to_index (dict): ラベルとインデックスのマッピング
    
    Returns:
        tuple: エンコードされたラベル、ラベルとインデックスのマッピング
    """
    if decoding:
        return[[k] for sublist in labels for i, x in enumerate(sublist) for k in label_to_index if x == 1 and i == label_to_index[k]]
    str_labels = [label for label in sum(labels, []) if type(label) == str]
    label_to_index = {label: idx for idx, label in enumerate(sorted(set(str_labels)))}
    zeros = [0] * len(label_to_index)
    for i in range(len(labels)):
        for j in range(len(labels[i])):
            if labels[i][j] in label_to_index and type(labels[i][j]) == str:
                labels[i][j] = zeros[:label_to_index[labels[i][j]]] + [1] + zeros[label_to_index[labels[i][j]]+1:]
    return [[x for y in sublist for x in (y if isinstance(y, list) else [y])] for sublist in labels], label_to_index

def split_dataset(X, y, train_size=0.8):  # データセットを学習用とテスト用に分割
    """
    データセットを学習用とテスト用に分割する関数

    Args:
        X (list): 入力
        y (list): 正解ラベル
        train_size (float): 学習データの割合
    
    Returns:
        tuple: 学習用データとテスト用データ
    """
    n = len(X)
    indices = list(range(n))
    random.shuffle(indices)
    X_train, y_train = [X[i] for i in indices[:int(n*train_size)]], [y[i] for i in indices[:int(n*train_size)]]
    X_test, y_test = [X[i] for i in indices[int(n*train_size):]], [y[i] for i in indices[int(n*train_size):]]
    return X_train, y_train, X_test, y_test

# データセット
# X = [[0, 0], [0, 1], [1, 0], [1, 1]]  # 入力
# y = [[1], [0], [0], [1]]  # 出力

# X = [[random.random(), random.random()] for _ in range(100)]
# y = [[1] if x[0] + x[1] > 1 else [0] for x in X]

# X = [[random.random(), random.random()] for _ in range(100)]
# y = [[x[0] + x[1]] for x in X]

# X = [[0, "A"], [1, "A"], [2, "B"], [3, "A"], [4, "B"], [5, "A"], [6, "B"], [7, "B"], [8, "A"], [9, "B"]]
# y = [[1], [1], [0], [1], [0], [1], [0], [0], [1], [0]]

X = [["A", "A"], ["A", "A"], ["B", "B"], ["A", "A"], ["B", "B"], ["A", "A"], ["B", "B"], ["B", "B"], ["A", "A"], ["B", "B"]]
y = [[1], [1], [0], [1], [0], [1], [0], [0], [1], [0]]

# X, min_val, max_val = normalize(X)
X, label_to_index = label_encoding(X)

X_train, y_train, X_test, y_test = split_dataset(X, y, train_size=0.8)

epochs = 1000  # エポック数
learning_rate = 0.01  # 学習率
layer_sizes = [len(X[0]), 8, 16, 8, len(y[0])]  # 各層のユニット数

hidden_activation = leaky_relu
output_activation = identity
loss = mean_absolute_error

# X_train: 入力, y_train: 出力, layer_sizes: 各層のユニット数, epochs: エポック数, learning_rate: 学習率, hidden_activation: 隠れ層の活性化関数, output_activation: 出力層の活性化関数, loss: 損失関数
weights, biases = train(X_train, y_train, layer_sizes, epochs, learning_rate, hidden_activation=hidden_activation, output_activation=output_activation, loss=loss)
# X_test: テスト入力, weights: 重み, biases: バイアス, hidden_activation: 隠れ層の活性化関数, output_activation: 出力層の活性化関数
predict_y = predict(X_test, weights, biases, hidden_activation=hidden_activation, output_activation=output_activation)

# X_test: テスト入力, y_test: テスト正解ラベル, predict_y: 予測値, loss: 損失関数
accuracy_num = accuracy(X_test, y_test, predict_y, loss=loss)