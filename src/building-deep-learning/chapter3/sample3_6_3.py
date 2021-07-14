import pickle
import numpy as np
from mnist import load_mnist    # mnistライブラリのload_mnist関数をインポート


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
    with open('sample_weight.pkl', 'rb') as f:
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

    sig = np.where(x < 0,
                   np.exp(x) / (1 + np.exp(x)),
                   1 / (1 + np.exp(-x)))
    return sig


def softmax(a):
    c = np.max(a)

    # オーバーフロー対策をしながら指数関数を計算する
    exp_a = np.exp(a - c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a

    return y


def predict(network: dict, x: np.ndarray):
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

    # 出力関数を通じて、10要素を得る
    y = softmax(a3)

    return y


def main():

    # 正規化されたテスト画像とテストラベルを取得
    test_images, expected_labels = get_normalized_test_data()

    # サンプルの重みとバイアスを取得
    sample_weight = get_sample_weight()

    # 的中した回数
    accuracy_cnt = 0

    # バッチサイズ
    batch_size = 100

    # 0〜test_imagesの量だけ、batch_size分だけスキップしたindexを作成
    for index in range(0, len(test_images), batch_size):

        # テストデータから、バッチサイズの入力を取得
        x_inputs = test_images[index:index+batch_size]

        # 予想を立てる
        y_outputs = predict(network=sample_weight, x=x_inputs)

        # 1次元目の要素を軸として、最も確率の高い要素のインデックス（推論された番号）を取得
        result_numbers = np.argmax(y_outputs, axis=1)

        # バッチサイズ分の期待値のリストを取得
        expecteds = expected_labels[index:index+batch_size]

        # NumPy配列同士で、演算子==により、boolean配列を作成
        result = (result_numbers == expecteds)

        # Trueの個数（推論が期待に合致した数）を求める
        accuracy_cnt += np.sum(result)

    # 予想の命中率を求める
    rate = accuracy_cnt / len(test_images)
    print("命中率:", rate)


if __name__ == '__main__':
    main()
