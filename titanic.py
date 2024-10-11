import pandas as pd
from sklearn.preprocessing import LabelEncoder

# データセットのパスを指定
train_path = '/Volumes/SSD/titanic/train.csv'
test_path = '/Volumes/SSD/titanic/test.csv'
# データセットを読み込む
train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

train = train.drop(columns=["Name", "PassengerId"])
test = test.drop(columns=["Name", "PassengerId"])
# ラベルエンコーダーを作成
label_encoder = LabelEncoder()

# PassengerId,Survived,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked
# ラベルエンコードする列を指定
columns_to_encode = ["Sex", "Ticket", "Cabin", "Embarked"]

# 各列をラベルエンコード
for column in columns_to_encode:
    train[column] = label_encoder.fit_transform(train[column].astype(str))
    test[column] = label_encoder.fit_transform(test[column].astype(str))

# 欠損値を各列の平均値で補完
train.fillna(train.mean(), inplace=True)
test.fillna(test.mean(), inplace=True)

from neural_networks import NeuralNetwork
from sklearn.neural_network import MLPClassifier

# nn = NeuralNetwork(hidden_size=[8, 16, 32, 64, 32, 16, 8], epochs=30, learning_rate=0.01)  # Create a neural network
# nn.train(train.drop(columns=["Survived"]).values.tolist(), [[y] for y in train["Survived"].values.tolist()])  # Train the neural network

# output = nn.predict(test.values.tolist())  # Predict the output
# print(output)

# 特徴量とターゲットを分ける
X_train = train.drop(columns=["Survived"])
y_train = train["Survived"]

# MLPClassifierのインスタンスを作成
mlp = MLPClassifier(hidden_layer_sizes=(8, 16, 32, 64, 32, 16, 8), max_iter=30, learning_rate_init=0.01)

# モデルを訓練
mlp.fit(X_train, y_train)

# テストデータで予測
predictions = mlp.predict(test)

print(predictions)