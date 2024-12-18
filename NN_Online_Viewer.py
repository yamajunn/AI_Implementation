import random
import time
import os

class Nn:
    def __init__(self, hidden_func="relu", output_func="sigmoid", loss_func="cross_entropy_loss", epochs=1000, learning_rate=0.01, random_state=None, layer_sizes=None):
        activate_functions = {"sigmoid": self.sigmoid, "relu": self.relu, "leaky_relu": self.leaky_relu, "identity": self.identity}
        loss_functions = {"cross_entropy_loss": self.cross_entropy_loss, "mean_squared_error": self.mean_squared_error, "mean_absolute_error": self.mean_absolute_error, "binary_cross_entropy": self.binary_cross_entropy_loss, "categorical_cross_entropy": self.categorical_cross_entropy_loss}
        self.hidden_func = activate_functions[hidden_func]
        self.output_func = activate_functions[output_func]
        self.loss_func = loss_functions[loss_func]
        self.epochs = epochs
        self.learning_rate = learning_rate
        if random_state is not None: random.seed(random_state)
        self.layer_sizes = layer_sizes
        self.napier_number = self.napiers_logarithm(100000000)
        self.tolerance = 1e-10  # sqrt許容誤差
        self.weights = []
        self.biases = []
        self.activations = []
    
    def napiers_logarithm(self, x):
        """
        ネイピア数を求める関数

        Args:
            x (float): 入力
        
        Returns:
            float: 出力
        """
        return (1 + 1 / x) ** x
    
    def ln(self, x, max_iter=20, tol=1e-12):
        """
        自然対数を求める関数

        Args:
            x (float): 入力
            max_iter (int): 最大反復回数
            tol (float): 許容誤差
        
        Returns:
            float: 出力
        
        Raises:
            ValueError: x が正でない場合
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
    
    def sqrt(self, x):
        """
        平方根を求める関数

        Args:
            x (float): 入力
        
        Returns:
            float: 出力
        """
        estimate = x / 2.0  # 初期推定値
        while True:
            new_estimate = (estimate + x / estimate) / 2  # ニュートン法による更新
            if abs(new_estimate - estimate) < self.tolerance:  # 収束したら終了
                return new_estimate
            estimate = new_estimate  # 推定値を更新


    def sigmoid(self, x, derivative=False):
        """
        Sigmoid 関数およびその微分

        Args:
            x (float): 入力
            derivative (bool): 微分を計算するかどうか
        
        Returns:
            float: 出力
        """
        if derivative:
            return self.sigmoid_derivative(x)
        return 1 / (1 + self.napier_number ** -x)

    def sigmoid_derivative(self, x):
        """
        Sigmoid 関数の微分

        Args:
            x (float): 入力
        
        Returns:
            float: 出力
        """
        return self.sigmoid(x) * (1 - self.sigmoid(x))


    def relu(self, x, derivative=False):
        """
        ReLU 関数およびその微分

        Args:
            x (float): 入力
            derivative (bool): 微分を計算するかどうか
        
        Returns:
            float: 出力
        """
        if derivative:
            return self.relu_derivative(x)
        return max(0, x)

    def relu_derivative(self, x):
        """
        ReLU 関数の微分

        Args:
            x (float): 入力
        
        Returns:
            float: 出力
        """
        return 1 if x > 0 else 0
    

    def leaky_relu(self, x, alpha=0.01, derivative=False):
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
            return self.leaky_relu_derivative(x, alpha)
        return x if x > 0 else alpha * x

    def leaky_relu_derivative(self, x, alpha=0.01):
        """
        Leaky ReLU 関数の微分

        Args:
            x (float): 入力
            alpha (float): ハイパーパラメータ
        
        Returns:
            float: 出力
        """
        return 1 if x > 0 else alpha
    

    def identity(self, x, derivative=False):
        """
        恒等関数およびその微分

        Args:
            x (float): 入力
            derivative (bool): 微分を計算するかどうか
        
        Returns:
            float: 出力
        """
        if derivative:
            return self.identity_derivative(x)
        return x

    def identity_derivative(self, x):
        """
        恒等関数の微分

        Args:
            x (float): 入力(未使用)
        
        Returns:
            int: 出力
        """
        return 1
    

    def cross_entropy_loss(self, y_true, y_pred):
        """
        交差エントロピー損失関数

        Args:
            y_true (list): 正解ラベル
            y_pred (list): 予測ラベル
        
        Returns:
            float: 出力
            
        Raises:
            ValueError: 入力リストの長さが異なる場合
        """
        if len(y_true) != len(y_pred): raise ValueError("Input lists must have the same length.")
        return -sum([t * self.ln(p + 1e-9) for t, p in zip(y_true, y_pred)])
    
    def mean_squared_error(self, y_true, y_pred):
        """
        平均二乗誤差を求める関数

        Args:
            y_true (list): 正解ラベル
            y_pred (list): 予測ラベル
        
        Returns:
            float: 出力
            
        Raises:
            ValueError: 入力リストの長さが異なる場合
        """
        if len(y_true) != len(y_pred): raise ValueError("Input lists must have the same length.")
        return sum([(t - p) ** 2 for t, p in zip(y_true, y_pred)]) / len(y_true)
    
    def mean_absolute_error(self, y_true, y_pred):
        """
        平均絶対誤差を求める関数

        Args:
            y_true (list): 正解ラベル
            y_pred (list): 予測ラベル
        
        Returns:
            float: 出力
        
        Raises:
            ValueError: 入力リストの長さが異なる場合
        """
        if len(y_true) != len(y_pred): raise ValueError("Input lists must have the same length.")
        return sum([abs(t - p) for t, p in zip(y_true, y_pred)]) / len(y_true)
    
    def binary_cross_entropy_loss(self, y_true, y_pred):
        """
        バイナリ交差エントロピー損失関数

        Args:
            y_true (list): 正解ラベル
            y_pred (list): 予測ラベル
        
        Returns:
            float: 出力
        
        Raises:
            ValueError: 入力リストの長さが異なる場合
        """
        if len(y_true) != len(y_pred): raise ValueError("Input lists must have the same length.")
        epsilon = 1e-9  # 0で割るのを防ぐための小さな値
        return -sum([t * self.ln(p + epsilon) + (1 - t) * self.ln(1 - p + epsilon) for t, p in zip(y_true, y_pred)]) / len(y_true)
    
    def categorical_cross_entropy_loss(self, y_true, y_pred):
        """
        カテゴリカル交差エントロピー損失関数

        Args:
            y_true (list): 正解ラベル
            y_pred (list): 予測ラベル
        
        Returns:
            float: 出力
        
        Raises:
            ValueError: 入力リストの長さが異なる場合
        """
        if len(y_true) != len(y_pred): raise ValueError("Input lists must have the same length.")
        epsilon = 1e-9  # 0で割るのを防ぐための小さな値
        return -sum([t * self.ln(p + epsilon) for t, p in zip(y_true, y_pred)]) / len(y_true)
    
    def initialize_weights(self):  # 重みとバイアスの初期化
        """
        重みとバイアスを初期化する関数
        """
        for i in range(len(self.layer_sizes) - 1):
            limit = self.sqrt(6 / (self.layer_sizes[i] + self.layer_sizes[i+1]))  # 重みの初期化に使う乱数の範囲
            self.weights.append([[random.uniform(-limit, limit) for _ in range(self.layer_sizes[i])] for _ in range(self.layer_sizes[i+1])])  # 重みは -limit から limit の間の乱数で初期化
            self.biases.append([0 for _ in range(self.layer_sizes[i+1])])  # バイアスは0で初期化
    
    def forward_propagation(self, inputs):  # 順伝播処理
        """
        順伝播処理を行う関数

        Args:
            inputs (list): 入力
        """
        self.activations = [inputs]
        for W, b in zip(self.weights, self.biases):
            z = [
                sum([self.activations[-1][i] * W[j][i] for i in range(len(self.activations[-1]))]) + b[j]
                for j in range(len(b))
            ]
            if W != self.weights[-1]:
                self.activations.append([self.hidden_func(z_i, derivative=False) for z_i in z])
            else:
                self.activations.append([self.output_func(z_i, derivative=False) for z_i in z])

    def backward_propagation(self, y_true):  # 逆伝播処理
        """
        逆伝播処理を行う関数

        Args:
            y_true (list): 正解ラベル
        """
        output_layer = self.activations[-1]
        errors = [
            (output_layer[i] - y_true[i]) * self.output_func(output_layer[i], derivative=True)
            for i in range(len(y_true))
        ]
        deltas = [errors]
        # 隠れ層の誤差を計算
        for l in range(len(self.weights)-1, 0, -1):
            hidden_errors = [
                sum([deltas[0][k] * self.weights[l][k][j] for k in range(len(deltas[0]))]) * self.hidden_func(self.activations[l][j], derivative=True)
                for j in range(len(self.activations[l]))
            ]
            deltas.insert(0, hidden_errors)
        # 重みとバイアスを更新
        for l in range(len(self.weights)):
            for i in range(len(self.weights[l])):
                for j in range(len(self.weights[l][i])):
                    self.weights[l][i][j] -= self.learning_rate * deltas[l][i] * self.activations[l][j]
                self.biases[l][i] -= self.learning_rate * deltas[l][i]
    
    def weight_view(self, X):
        os.system('cls' if os.name == 'nt' else 'clear')
        for i, layer in enumerate(self.weights):
            print(f"Layer {i+1} weights:")
            for neuron_weights in layer:
                print(neuron_weights)
            print(f"Inputs: {self.activations[-2]}")
            print(f"Outputs: {self.activations[-1]}")
            print()
        time.sleep(0.1)
        
    def fit(self, X, y):  # 学習
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
        self.initialize_weights()
        start = time.time()
        for epoch in range(self.epochs):
            total_loss = 0
            for i in range(len(X)):
                self.forward_propagation(X[i])
                self.weight_view(X[i])
                total_loss += self.loss_func(y[i], self.activations[-1])
                self.backward_propagation(y[i])

        print(f"Training time: {time.time()-start:.2f} seconds")
        
    def predict(self, X):  # 予測
        """
        予測を行う関数

        Args:
            X (list): 入力
        
        Returns:
            list: 出力
        """
        outputs = []
        for i in range(len(X)):  # Prediction
            self.forward_propagation(X[i])
            outputs.append(self.activations[-1])
        return outputs
    
    def accuracy(self, X, y, predict):  # 予測精度の計算
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
            total_loss += self.loss_func(y[i], predict[i])
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

def split_dataset(X, y, train_size=0.8, random_state=None):  # データセットを学習用とテスト用に分割
    """
    データセットを学習用とテスト用に分割する関数

    Args:
        X (list): 入力
        y (list): 正解ラベル
        train_size (float): 学習データの割合
    
    Returns:
        tuple: 学習用データとテスト用データ
    """
    if random_state is not None: random.seed(random_state)
    n = len(X)
    indices = list(range(n))
    random.shuffle(indices)
    X_train, y_train = [X[i] for i in indices[:int(n*train_size)]], [y[i] for i in indices[:int(n*train_size)]]
    X_test, y_test = [X[i] for i in indices[int(n*train_size):]], [y[i] for i in indices[int(n*train_size):]]
    return X_train, y_train, X_test, y_test

X = [[0, 0], [0, 1], [1, 0], [1, 1]]  # 入力
y = [[1], [0], [0], [1]]  # 出力

# X = [[random.random(), random.random()] for _ in range(100)]
# y = [[x[0] + x[1]] for x in X]

X_train, y_train, X_test, y_test = split_dataset(X, y, train_size=0.5, random_state=42)  # データセットを学習用とテスト用に分割

epochs = 20  # エポック数
learning_rate = 0.01  # 学習率
layer_sizes = [len(X[0]), 4, 4, len(y[0])]  # 各層のユニット数

hidden_activation = "relu"
output_activation = "identity"
loss = "mean_squared_error"

random_state = 42  # 乱数シード

nn = Nn(hidden_activation, output_activation, loss, epochs, learning_rate, random_state, layer_sizes)
nn.fit(X_train, y_train)
predictions = nn.predict(X_test)
# nn.accuracy(X_test, y_test, predictions)