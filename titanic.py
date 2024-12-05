import pandas as pd
from sklearn.preprocessing import LabelEncoder

# データセットのパスを指定
csv_path = '/Users/chinq500/Desktop/Programs/AI/titanic/train.csv'

# データセットを読み込む
data = pd.read_csv(csv_path)

data = data.drop(columns=["Name", "PassengerId"])
# ラベルエンコーダーを作成
label_encoder = LabelEncoder()

# PassengerId,Survived,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked
# ラベルエンコードする列を指定
columns_to_encode = ["Sex", "Ticket", "Cabin", "Embarked"]

# 各列をラベルエンコード
for column in columns_to_encode:
    data[column] = label_encoder.fit_transform(data[column].astype(str))

# 欠損値を各列の平均値で補完
data.fillna(data.mean(), inplace=True)

from neural_networks import NeuralNetwork
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# nn = NeuralNetwork(hidden_size=[8, 16, 32, 64, 32, 16, 8], epochs=30, learning_rate=0.01)  # Create a neural network
# nn.train(train.drop(columns=["Survived"]).values.tolist(), [[y] for y in train["Survived"].values.tolist()])  # Train the neural network

# output = nn.predict(test.values.tolist())  # Predict the output
# print(output)

# 特徴量とターゲットを分ける
X = data.drop(columns=["Survived"])
y = data["Survived"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# MLPClassifierのインスタンスを作成
mlp = MLPClassifier(hidden_layer_sizes=(8, 16, 32, 64, 32, 16, 8), max_iter=30, learning_rate_init=0.01)

# モデルを訓練
mlp.fit(X_train, y_train)

# テストデータで予測
predictions = mlp.predict(X_test)

print(predictions)

# 精度
accuracy = accuracy_score(y_test, predictions)

print(f'Accuracy: {accuracy * 100:.2f}%')
