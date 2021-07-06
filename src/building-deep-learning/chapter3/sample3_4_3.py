import numpy as np
import matplotlib.pylab as plotlib
from sample3_2_4 import sigmoid


# 0. スケッチでニューラルネットワークの設計図を描く

def init_network():
    """
    重みとバイアスのネットワークを初期化して返す
    """

    network = {}

    # ----- 第1層への重みとバイアスを宣言する-----
    # W1: 2x3
    network['W1'] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
    # b1: 3x1
    network['b1'] = np.array([0.1, 0.2, 0.3])

    # ----- 第2層への重みとバイアスを宣言する-----
    # W2: 3x2
    network['W2'] = np.array([[0.1, 0.4], [0.2, 0.4], [0.5, 0.6]])
    # b2: 2x1
    network['b2'] = np.array([0.1, 0.2])

    # ----- 第3層への重みとバイアスを宣言する-----
    # W3: 2x2
    network['W3'] = np.array([[0.1, 0.3], [0.2, 0.4]])
    # b3: 2x1
    network['b3'] = np.array([0.1, 0.2])

    return network


def identity_function(x):
    """ 恒等関数 """
    return x


def forward(network, x):
    """ 入力信号を、出力に変換する """

    # 各層の重みを取得する
    W1, W2, W3 = network['W1'], network['W2'], network['W3']

    # 各層へのバイアスを取得する
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    # 入力を第1層に伝達する
    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)

    # 第1層の結果を、第2層に伝達する
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)

    # 第2層の結果を、第3層に伝達する
    a3 = np.dot(z2, W3) + b3

    # 恒等関数を経由して、出力を生成する
    y = identity_function(a3)

    return y



def main():
    # ニューラルネットワークの初期データを取得する
    network = init_network()

    # 入力を取得する
    x = np.array([1, 5])

    # 出力を取得する
    y = forward(network, x)

    print(y)


if __name__ == '__main__':
    main()
