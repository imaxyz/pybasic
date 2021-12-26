import numpy as np
from layers import *
from mnist import *


class SGD:
    """確率的勾配降下法（Stochastic Gradient Descent）"""

    def __init__(self, lr=0.01):
        """
        初期化
        :param lr: learning rate(学習係数)
        """

        # 学習係数をインスタンスとして保持する
        self.lr = lr

    def update_params(self, params: dict, grads: dict):
        """
        SGDでは、このupdateメソッドを繰り返し実行する。

        :param params: 重みのディクショナリ変数
        :param grads: 勾配のディクショナリ変数
        :return:
        """
        for key in params.keys():
            params[key] -= self.lr * grads[key]


def main():
    # 2層ニューラルネットワークを生成
    network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

    # 最適化を行うオブジェクトを生成
    optimizer = SGD()

    # データの読み込み
    (x_train, t_train), (x_test, t_test) = load_mnist(
        # 画像ピクセル値を0〜1.0に正規化する
        normalize=True,
        # one-hot表現で取得する
        one_hot_label=True
    )

    # 訓練データの母数を取得 ... mnistは6万枚
    train_size = x_train.shape[0]
    # バッチサイズを定義
    batch_size = 100

    for i in range(100):

        # 毎回6万枚の訓練データから、ランダムに100枚のデータを抜き出す
        batch_mask = np.random.choice(train_size, batch_size)

        # バッチ用の訓練画像データを取得
        x_batch = x_train[batch_mask]

        # バッチ用の訓練ラベルを取得
        t_batch = t_train[batch_mask]

        params = network.params
        gradient = network.gradient(x=x_batch, t=t_batch)

        optimizer.update_params(params=params, grads=gradient)
        pass

    pass


if __name__ == '__main__':
    main()


