import pickle
import sys, os
import numpy as np
from mnist import load_mnist    # mnistライブラリのload_mnist関数をインポート
from PIL import Image


def get_normalized_test_data():
    """正規化されたテストデータを返す"""

    (train_images, train_labels), (test_images, test_labels) = load_mnist(
        normalize=True,
        flatten=True,
        one_hot_label=False,
    )

    return test_images, test_labels


def get_sample_weight():
    """サンプルの重みを返す"""

    # sample_weight.pkl ... 重みWとバイアスbのパラメータがディクショナリ型で登録されている
    with open('sample_weight.pkl') as f:
        network = pickle.load(f)

    return network


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def stable_sigmoid(x):
    """
    安定したシグモイド関数
    https://www.delftstack.com/ja/howto/python/sigmoid-function-python/

    :param x:
    :return:
    """

    sig = np.where(x < 0, np.exp(x) / (1 + np.exp(x)), 1 / (1 + np.exp(-x)))
    return sig


def softmax(a):
    c = np.max(a)

    # オーバーフロー対策をしながら指数関数を計算する
    exp_a = np.exp(a - c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a

    return y


def predict(network, x):
    """
    3層ニューラルネットワークで予測する

    設計:
        - 入力層: 784
        - 第1隱れ層: 50
        - 第2隱れ層: 100
        - 出力層: 10

    :param network:
        ネットワーク情報（重み、バイアス）
    :param x:
        入力
    :return:
    """

    # 各層の重みを取得
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = stable_sigmoid(a1)

    a2 = np.dot(z1, W2) + b2
    z2 = stable_sigmoid(a2)

    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)

    return y


def main():

    sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定

    test_images, test_labels = get_normalized_test_data()

    sample_weight = get_sample_weight()

    accuracy_cnt = 0
    for i in range(len(test_images)):
        y = predict(sample_weight, test_images[i])
        p = np.argmax(y)  # 最も確率の高い要素のインデックスを取得
        if p == test_labels[i]:
            accuracy_cnt += 1

    print("Accuracy:" + str(float(accuracy_cnt) / len(test_images)))


if __name__ == '__main__':
    main()
