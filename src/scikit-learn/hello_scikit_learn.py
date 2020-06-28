from sklearn import datasets
from sklearn import tree
from sklearn.metrics import accuracy_score


def divide_training_data(digits):
    """トレーニングデータと、テストデータに分割する"""

    # トレーニングデータ / テストデータの区切り
    train_size = int(len(digits.data) * 4 / 5)

    # トレーニングデータ
    train_data = digits.data[:train_size]

    # トレーニングデータの正解
    train_label = digits.target[:train_size]

    # テストデータ
    test_data = digits.data[train_size:]

    # テストデータの正解
    test_label = digits.target[train_size:]

    return train_data, train_label, test_data, test_label


# 機械学習アルゴリズムの決定木(学習済みモデル)を生成
model = tree.DecisionTreeClassifier()

# MINISTデータのダウンロード
digits = datasets.load_digits()
train_data, train_label, test_data, test_label = divide_training_data(
    digits=digits)

# 機械学習を実行して、学習済みモデルを作成
model.fit(train_data, train_label)

# 正解の予測と、実際の正解を比較
predicted = model.predict(test_data)

print('正確さ:', accuracy_score(test_label, predicted))
print('Done!')
