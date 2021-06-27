import numpy as np
import matplotlib.pylab as plotlib
from sample3_2_4 import sigmoid


def identity_function(x):
    """恒等関数"""
    return x


def main():
    # ----------入力〜第1層----------
    # 第1層へのバイアス
    B1 = np.array([0.1, 0.2, 0.3])

    # 入力(2つのニューロン)
    X = np.array([1.0, 0.5])

    # 入力に対する重み
    W1 = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])

    # 第1層の重みつき和
    A1 = np.dot(X, W1) + B1

    # 活性化関数を経由した、第1層の値を求める
    # [ 0.57444252  0.66818777  0.75026011]
    Z1 = sigmoid(A1)
    print(Z1)

    # ----------第1層〜第2層----------
    W2 = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
    B2 = np.array([0.1, 0.2])
    A2 = np.dot(Z1, W2) + B2
    Z2 = sigmoid(A2)

    # [ 0.62624937  0.7710107 ]
    print(Z2)

    # ----------第2層〜出力----------
    W3 = np.array([[0.1, 0.3], [0.2, 0.4]])
    B3 = np.array([0.1, 0.2])
    A3 = np.dot(Z2, W3) + B3
    Z3 = sigmoid(A3)

    # 恒等関数を経由して出力する
    Z3_2 = identity_function(Z3)

    # [ 0.62624937  0.7710107 ]
    print(Z3_2)


if __name__ == '__main__':
    main()
