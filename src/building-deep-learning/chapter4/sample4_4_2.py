# coding: utf-8
import sys, os

sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定
import numpy as np


def numerical_gradient(func, x: np.ndarray):
    """多次元配列に対応した勾配を求める"""

    # 勾配を初期化する
    gradient = np.zeros_like(x)

    # 入力xに対するイテレーターを生成する
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])

    h = 1e-4  # 0.0001

    # 入力xの全要素分、処理を繰り返す
    while not it.finished:

        # 要素のインデックス(タプル形式) を取得
        element_index_tuple = it.multi_index

        # 入力された関数で数値微分する
        tmp_val = x[element_index_tuple]
        x[element_index_tuple] = tmp_val + h
        fxh1 = func(x)  # f(x+h)

        x[element_index_tuple] = tmp_val - h
        fxh2 = func(x)  # f(x-h)

        # 微分の結果を、勾配リストに登録する
        gradient[element_index_tuple] = (fxh1 - fxh2) / (2 * h)

        x[element_index_tuple] = tmp_val  # 値を元に戻す

        it.iternext()

    return gradient


class SimpleNet:

    def __init__(self):
        # 正規分布(ガウス分布)を持つ2x3の乱数行列を作成して、ダミーの重みとする
        self.W = np.random.randn(2, 3)
        pass

    def _predict(self, input_x: np.ndarray):
        """
        入力されたパラメータ配列と、重みの、行列積を返す
        :param input_x: 入力された行列
        :return: 処理結果の行列積
        """

        result = np.dot(input_x, self.W)
        return result

    @staticmethod
    def _softmax(input_x: np.ndarray):
        """
        ソフトマックス関数(出力関数)
        入力を結果が0〜1の間の実数に変換する
        """
        revised = input_x - np.max(input_x, axis=-1, keepdims=True)  # オーバーフロー対策
        result = np.exp(revised) / np.sum(np.exp(revised), axis=-1, keepdims=True)

        return result

    @staticmethod
    def _one_hot_cross_entropy_error(y: np.ndarray, t: np.ndarray):
        """
        交差エントロピー誤差を返す

        :param y: ニューラルネットワークの出力
        :param t: one-hot表現の教師データ
        :return: 損失関数の計算結果
        """
        if y.ndim == 1:
            # ニューラルネットワークが1次元データ(n, )の場合、バッチ処理用に(1, n)へ整形する
            t = t.reshape(1, t.size)
            y = y.reshape(1, y.size)

        batch_size = y.shape[0]

        # 微小の値: deltaを宣言
        # 1e-2: 1 x 10**-2 = 0.01
        # 1e-7: 1 x 10**-7 = 0.0000001
        delta = 1e-7

        # np.log(0)の時、マイナスの無限大-infになり、暴走してしまうので、deltaを加える
        # result = -np.sum(t * np.log(y + delta)) ...0.7339688834135556
        result = -np.sum(t * np.log(y + delta)) / batch_size
        return result

    def loss(self, input_x: np.ndarray, input_t: np.ndarray):
        """
        損失関数の値を求める

        :param input_x: 入力データ（行）
        :param input_t: 正解ラベル（列）
        :return:
        """

        # 入力データと重みの行列積を求める
        z = self._predict(input_x)

        # 出力関数として、ソフトマックス関数にかける
        y = SimpleNet._softmax(z)

        # 損失関数の値を求める: 交差エントロピー誤差を計算する。結果が小さい方が良い
        loss_value = SimpleNet._one_hot_cross_entropy_error(y=y, t=input_t)

        return loss_value


def main():
    """ニューラルネットワークの勾配を求める"""

    # 2行分のデータ
    x = np.array([0.6, 0.9])

    # 正解ラベル(3列分)
    t = np.array([0, 0, 1])

    # ニューラルネットワークの勾配を求めるオブジェクトを生成
    net = SimpleNet()

    # 損失関数(交差エントロピー誤差を使用)を定義
    # --> 重みwを引数に取り、net.loss(x, t) の結果を返す無名関数を定義している
    func = lambda w: net.loss(x, t)

    # 重みパラメータに関数する損失関数の勾配を求める（多次元配列に対応）
    # 結果(dW) は、重みWと同じ形状になる
    dW = numerical_gradient(func, net.W)
    print(dW)


if __name__ == '__main__':
    main()

