# coding: utf-8
import sys, os

sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定
import numpy as np


def numerical_gradient(func, x: np.ndarray):
    """多次元配列に対応した勾配を求める"""
    print('numerical_gradient(), x shape: ', x.shape)

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
        value = (fxh1 - fxh2) / (2 * h)
        gradient[element_index_tuple] = value

        x[element_index_tuple] = tmp_val  # 値を元に戻す

        it.iternext()

    return gradient


class TwoLayerNet:
    """2層ニューラルネットワーク"""

    def __init__(self,
                 input_size,
                 hidden_size,
                 output_size,
                 weight_init_std=0.01):
        """
        重みパラメータを初期化する

        :param input_size: 入力層のニューロン数
        :param hidden_size: 隱れ層のニューロン数
        :param output_size: 出力層のニューロン数
        :param weight_init_std:
        """

        # params: このネットワークに必要なパラメータを集約する
        self.params = {}

        # W1: 1層目の重み。正規分布に従う乱数で初期化する
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)

        # b1: 1層目のバイアスをゼロで初期化する
        self.params['b1'] = np.zeros(hidden_size)

        # W2: 2層目の重み
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)

        # b2: 2層目のバイアスをゼロで初期化する
        self.params['b2'] = np.zeros(output_size)

    def _sigmoid(self, x: np.ndarray):
        return 1 / (1 + np.exp(-x))

    def _softmax(self, x: np.ndarray):
        x = x - np.max(x, axis=-1, keepdims=True)  # オーバーフロー対策
        return np.exp(x) / np.sum(np.exp(x), axis=-1, keepdims=True)

    def _predict(self, x: np.ndarray):
        """
        推論（認識）を行う
        入力されたパラメータ配列と、重みの、行列積を返す

        :param x: 画像データ
        :return:
        """

        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']

        # 1層目の入力信号の総和を求める
        a1 = np.dot(x, W1) + b1
        # 活性化関数にシグモイド関数を用いて1層目の値を0.0〜1.0に丸める
        z1 = self._sigmoid(a1)

        # 2層目の入力信号の総和を求める
        a2 = np.dot(z1, W2) + b2
        # 出力関数にソフトマックス関数を用いて2層目の値を0.0〜1.0に丸める
        y = self._softmax(a2)

        return y

    def _label_cross_entropy_error(self, y, t):
        """
        1個あたり平均の交差エントロピー誤差を返す

        :param y:
        :param t: ラベル表現の訓練データ
        :return:
        """
        if y.ndim == 1:
            t = t.reshape(1, t.size)
            y = y.reshape(1, y.size)

        # 教師データがone-hot-vectorの場合、正解ラベルのインデックスに変換
        if t.size == y.size:
            t = t.argmax(axis=1)

        batch_size = y.shape[0]

        result = -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size
        return result

    def loss(self, x, t):
        """
        損失関数の値を求める

        :param x: 入力データ
        :param t: 教師データ
        :return:
        """
        # ニューラルネットワークで伝搬する
        y = self._predict(x)

        # 伝搬結果の損失確率を求める
        loss_result = self._label_cross_entropy_error(y, t)

        return loss_result

    def accuracy(self, x, t):
        """
        ニューラルネットワークの精度を求める
        :param x:
        :param t:
        :return:
        """
        # 入力データに対して、ネットワークの重みとバイアスを考慮してニューラルネットワークの計算を行う
        y = self._predict(x=x)

        # ニューラルネットワーク予測の各行における最大値のインデックスを求める
        y_max_indexes = np.argmax(y, axis=1)

        # 訓練データの各行における最大値のインデックスを求める
        t_max_indexes = np.argmax(t, axis=1)

        # 正答率を計算する
        accuracy = np.sum(y_max_indexes == t_max_indexes) / float(x.shape[0])
        return accuracy

    def get_numerical_gradient(self, x, t):
        """
        数値微分により重みに対する勾配を求める（低速版）

        :param x: 入力データ
        :param t: 教師データ
        :return:
        """

        # 損失関数を定義
        loss_W = lambda W: self.loss(x, t)

        # 1層目の重みの勾配を求める
        print('start numerical_gradient(W1)...')
        W1 = numerical_gradient(loss_W, self.params['W1'])

        # 1層目のバイアスの勾配を求める
        print('start numerical_gradient(b1)...')
        b1 = numerical_gradient(loss_W, self.params['b1'])

        # 2層目の重みの勾配を求める
        print('start numerical_gradient(W2)...')
        W2 = numerical_gradient(loss_W, self.params['W2'])

        # 1層目のバイアスの勾配を求める
        print('start numerical_gradient(b2)...')
        b2 = numerical_gradient(loss_W, self.params['b2'])

        # 各種勾配をまとめて返す
        return { 'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2 }

    def _sigmoid_grad(self, x):
        return (1.0 - self._sigmoid(x)) * self._sigmoid(x)

    def gradient(self, x, t):
        """
        誤差逆伝搬法により重みに対する勾配を求める（高速版）

        :param x:
        :param t:
        :return:
        """
        # ニューラルネットワークの計算を実行する。
        # _predict()との違いは、a1, z1を後続の処理で使用すること
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']

        # forward
        a1 = np.dot(x, W1) + b1
        z1 = self._sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = self._softmax(a2)

        # backward
        batch_num = x.shape[0]
        grads = {}
        dy = (y - t) / batch_num
        grads['W2'] = np.dot(z1.T, dy)
        grads['b2'] = np.sum(dy, axis=0)

        dz1 = np.dot(dy, W2.T)
        da1 = self._sigmoid_grad(a1) * dz1
        grads['W1'] = np.dot(x.T, da1)
        grads['b1'] = np.sum(da1, axis=0)

        return grads


def main():

    # 画像サイズを定義(mnistを想定)
    image_size = 28*28

    # 2層ニューラルネットワークを生成する
    net = TwoLayerNet(input_size=image_size,    # 入力層のニューロン数
                      output_size=10,           # 出力層のニューロン数
                      hidden_size=100)          # 隠れ層のニューロン数

    # ダミーの入力データを作成（画像100枚分）
    x = np.random.rand(100, image_size)

    # ダミーの正解ラベルを生成（画像100枚分）
    t = np.random.rand(100, 10)

    # ニューラルネットワークの精度を求める
    accuracy = net.accuracy(x=x, t=t)
    print('accuracy: ', accuracy)

    # 数値微分で入力データの勾配を求める
    gradient = net.get_numerical_gradient(x, t)

    # 誤差逆伝搬法で入力データの勾配を求める
    # gradient = net.gradient(x, t)
    print('gradient.keys: ', gradient.keys())

    # 損失関数の値を計算する
    loss_result = net.loss(x=x, t=t)
    print('loss result: ', loss_result)

    pass

if __name__ == '__main__':
    # main()
    main()


