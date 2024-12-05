# %% [markdown]
# モジュールをインポート

# %%
import random
import time

# %% [markdown]
# ### ネイピア数
# $$e = \lim_{{x \to \infty}} \left(1 + \frac{1}{x}\right)^x$$

# %%
def napiers_logarithm(x):
    return (1 + 1 / x) ** x
napier_number = napiers_logarithm(100000000)  # e

# %% [markdown]
# ### シグモイド関数
# $$Sigmoid(x) = \frac{1}{1 + e^{-x}}$$

# %%
def sigmoid(x):
    return 1 / (1 + napier_number ** -x)

# %% [markdown]
# ### シグモイド関数の微分
# $$Sigmoid'(x) = Sigmoid(x) \cdot (1 - Sigmoid(x))$$

# %%
def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

# %% [markdown]
# ### ReLU関数
# $$ReLU(x) = \max(0, x)$$

# %%
def relu(x):
    return max(0, x)

# %% [markdown]
# ### ReLU関数の微分
# $$ReLU'(x) = \begin{cases} 
# 1 & (x > 0) \\
# 0 & (x ≤ 0)
# \end{cases}$$

# %%
def relu_derivative(x):
    return 1 if x > 0 else 0

# %% [markdown]
# ### 自然対数
# 1. **スケーリング**  
#    自然対数の性質 $\ln(a \cdot b) = \ln(a) + \ln(b)$ を利用して、$x$ を 1 に近い値に変換
# 
#    $$
#    k = 0, \quad x > 2 \text{ の間 } x = \frac{x}{2}, \, k += 1
#    $$
# 
#    $$
#    x < 0.5 \text{ の間 } x = 2 \cdot x, \, k -= 1
#    $$
# 
#    変換後、$x \in [0.5, 2]$ に収まる
# 
# 2. **ニュートン法**  
#    方程式 $f(y) = e^y - x = 0$ を解くため、ニュートン法を適用
# 
#    $$
#    y_{n+1} = y_n - \frac{e^{y_n} - x}{e^{y_n}}
#    $$
# 
#    初期値として $y_0 = x - 1$ を用いる。
# 
# 3. **結果**  
#    最終的な計算：
# 
#    $$
#    \ln(x) = y + k \cdot \ln(2)
#    $$
# 
#    $\ln(2) \approx 0.6931471805599453$ を用いる

# %%
def ln(x, max_iter=20, tol=1e-12):
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

# %% [markdown]
# ### クロスエントロピー損失
# $$ L = -\sum_{i=1}^{N} y_i \cdot \ln(\hat{y}_i + \epsilon)$$

# %%
def cross_entropy_loss(y_true, y_pred):
    if len(y_true) != len(y_pred): raise ValueError("Input lists must have the same length.")
    return -sum([t * ln(p + 1e-9) for t, p in zip(y_true, y_pred)])

# %% [markdown]
# ### 平方根を計算

# %%
def sqrt(x):
    tolerance = 1e-10  # 許容誤差
    estimate = x / 2.0  # 初期推定値
    while True:
        new_estimate = (estimate + x / estimate) / 2  # ニュートン法による更新
        if abs(new_estimate - estimate) < tolerance:  # 収束したら終了
            return new_estimate
        estimate = new_estimate  # 推定値を更新

# %% [markdown]
# ### ニューラルネットワークを初期化

# %%
def initialize_weights(layer_sizes):
    weights, biases = [], []
    for i in range(len(layer_sizes) - 1):
        limit = sqrt(6 / (layer_sizes[i] + layer_sizes[i+1]))  # 重みの初期化に使う乱数の範囲
        weights.append([[random.uniform(-limit, limit) for _ in range(layer_sizes[i])] for _ in range(layer_sizes[i+1])])  # 重みは -limit から limit の間の乱数で初期化
        biases.append([0 for _ in range(layer_sizes[i+1])])  # バイアスは0で初期化
    return weights, biases

# %% [markdown]
# ### 順伝播処理

# %%
def forward_propagation(inputs, weights, biases):  # 順伝播処理
    activations = [inputs]
    for W, b in zip(weights, biases):
        z = [
            sum([activations[-1][i] * W[j][i] for i in range(len(activations[-1]))]) + b[j]
            for j in range(len(b))
        ]
        if W != weights[-1]:
            activations.append([relu(z_i) for z_i in z])
        else:
            activations.append([sigmoid(z_i) for z_i in z])
    return activations

# %% [markdown]
# ### 逆伝播処理

# %%
def backward_propagation(activations, y_true, weights, biases, learning_rate):  # 逆伝播処理
    output_layer = activations[-1]
    errors = [
        (output_layer[i] - y_true[i]) * sigmoid_derivative(output_layer[i])
        for i in range(len(y_true))
    ]
    deltas = [errors]
    # 隠れ層の誤差を計算
    for l in range(len(weights)-1, 0, -1):
        hidden_errors = [
            sum([deltas[0][k] * weights[l][k][j] for k in range(len(deltas[0]))]) * relu_derivative(activations[l][j])
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

# %% [markdown]
# ### データをバッチサイズに分割

# %%
def create_batches(X, y, batch_size):  # ミニバッチを作成
    dataset = list(zip(X, y))  # 入力とラベルをまとめる
    random.shuffle(dataset)  # データセットをシャッフル
    batches = [
        dataset[i:i + batch_size] for i in range(0, len(dataset), batch_size)
    ]
    return batches

# %% [markdown]
# ### ミニバッチで学習

# %%
def train(X, y, layer_sizes, epochs, learning_rate, batch_size):  # ニューラルネットワークの学習
    weights, biases = initialize_weights(layer_sizes)
    start = time.time()
    
    for epoch in range(epochs):
        total_loss = 0
        batches = create_batches(X, y, batch_size)
        
        for batch in batches:
            batch_X = [b[0] for b in batch]
            batch_y = [b[1] for b in batch]
            
            # バッチごとに順伝播・逆伝播を行い、損失を計算
            activations_list = []
            for i in range(len(batch_X)):
                activations = forward_propagation(batch_X[i], weights, biases)
                activations_list.append(activations)
                total_loss += cross_entropy_loss(batch_y[i], activations[-1])
            
            # バッチ全体での逆伝播
            batch_gradients = [backward_propagation(activations_list[i], batch_y[i], weights, biases, learning_rate)
                            for i in range(len(batch_X))]
            
            # 平均勾配で重み・バイアスを更新
            for l in range(len(weights)):
                for i in range(len(weights[l])):
                    for j in range(len(weights[l][i])):
                        weights[l][i][j] -= learning_rate * sum(
                            grad[0][l][i][j] for grad in batch_gradients
                        ) / batch_size
                    biases[l][i] -= learning_rate * sum(
                        grad[1][l][i] for grad in batch_gradients
                    ) / batch_size

        m = epoch // (epochs // 20) + 1
        print(f"\rEpoch {epoch+1}/{epochs}, Loss: {total_loss/len(X):.10f}, [{'+'*m}{' '*(20-m)}]{' '*5}", end="")
    
    print(f"Training time: {time.time()-start:.2f} seconds")
    return weights, biases

# %% [markdown]
# ### 予測

# %%
def predict(X, weights, biases):  # 予測
    outputs = []
    for i in range(len(X)):  # Prediction
        outputs.append(forward_propagation(X[i], weights, biases)[-1])
    return outputs

# %% [markdown]
# ### 精度計算

# %%
def accuracy(X, y, predict):  # 予測精度の計算
    accuracy = 0
    for i in range(len(predict)):  # Prediction
        print(f"入力: {X[i]}, 正解: {y[i]}, 予測値: {[0 if p<0.5 else 1 for p in predict[i]]}, 出力地: {predict[i]}")
        accuracy += 1 if [0 if p<0.5 else 1 for p in predict[i]] == y[i] else 0
    print(f"正解率: {accuracy / len(predict):.2f}")
    return accuracy / len(predict)

# %% [markdown]
# ### データセット分割

# %%
def split_dataset(X, y, train_size=0.8):  # データセットを学習用とテスト用に分割
    n = len(X)
    indices = list(range(n))
    random.shuffle(indices)
    X_train, y_train = [X[i] for i in indices[:int(n*train_size)]], [y[i] for i in indices[:int(n*train_size)]]
    X_test, y_test = [X[i] for i in indices[int(n*train_size):]], [y[i] for i in indices[int(n*train_size):]]
    return X_train, y_train, X_test, y_test

# %% [markdown]
# ### メインコード

# %%
# データセット
# X = [[0, 0], [0, 1], [1, 0], [1, 1]]  # 入力
# y = [[1, 1], [1, 0], [0, 1], [0, 0]]  # 出力

X = [[random.random(), random.random()] for _ in range(10)]
y = [[1] if x[0] + x[1] > 1 else [0] for x in X]

X_train, y_train, X_test, y_test = split_dataset(X, y, train_size=0.8)

epochs = 5000  # エポック数
learning_rate = 0.01  # 学習率
layer_sizes = [2, 8, 16, 8, 1]  # 各層のユニット数
batch_size = 16  # ミニバッチのサイズ

# X_train: 入力, y_train: 出力, layer_sizes: 各層のユニット数, epochs: エポック数, learning_rate: 学習率, batch_size: バッチサイズ
weights, biases = train(X_train, y_train, layer_sizes, epochs, learning_rate, batch_size)

# X_test: テスト入力, weights: 重み, biases: バイアス
predict_y = predict(X_test, weights, biases)

# X_test: テスト入力, y_test: テスト正解ラベル, predict_y: 予測値
accuracy_num = accuracy(X_test, y_test, predict_y)


