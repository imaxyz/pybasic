from layers import *
from mnist import *
import numpy as np


def main():
    # データの読み込み
    (x_train, t_train), (x_test, t_test) = load_mnist(
        # 画像ピクセル値を0〜1.0に正規化する
        normalize=True,
        # one-hot表現で取得する
        one_hot_label=True
    )

    # 2層のニューラルネットワークを生成
    network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

    # 訓練データの一部を取得（リストの最初から3番目まで）
    x_batch = x_train[:3]
    t_batch = t_train[:3]

    # 数値微分により勾配を求める
    grad_numerical = network.numerical_gradient(x_batch, t_batch)

    # 誤差逆伝播法により勾配を求める
    grad_backprop = network.gradient(x_batch, t_batch)

    for key in grad_numerical.keys():
        # 誤差逆伝播法による勾配と、数値微分による勾配の差を求める
        diff = np.average(np.abs(grad_backprop[key] - grad_numerical[key]))
        print(key + ":" + str(diff))

    pass


if __name__ == '__main__':
    main()


