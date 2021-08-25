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

    count = 0
    # 入力xの全要素分、処理を繰り返す
    while not it.finished:
        # print('numerical_gradient() loop: ', count)

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
        count += 1

    return gradient


class SimpleNet:
    """単層ニューラルネットワーク"""

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

    def predict(self, x: np.ndarray):
        """
        推論（認識）を行う
        入力されたパラメータ配列と、重みの、行列積を返す

        :param x: 画像データ
        :return:
        """

        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']

        a1 = np.dot(x, W1) + b1
        z1 = self._sigmoid(a1)

        a2 = np.dot(z1, W2) + b2
        y = self._softmax(a2)

        return y

    def _cross_entropy_error(self, y, t):
        if y.ndim == 1:
            t = t.reshape(1, t.size)
            y = y.reshape(1, y.size)

        # 教師データがone-hot-vectorの場合、正解ラベルのインデックスに変換
        if t.size == y.size:
            t = t.argmax(axis=1)

        batch_size = y.shape[0]
        return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size

    def loss(self, x, t):
        """
        損失関数の値を求める

        :param x: 入力データ
        :param t: 教師データ
        :return:
        """
        y = self.predict(x)

        return self._cross_entropy_error(y, t)

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    def numerical_gradient(self, x, t):
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
        return {
            'W1': W1,
            'b1': b1,
            'W2': W2,
            'b2': b2
        }

    def sigmoid_grad(self, x):
        return (1.0 - self._sigmoid(x)) * self._sigmoid(x)

    def gradient(self, x, t):
        """
        誤差逆伝搬法により重みに対する勾配を求める（高速版）

        :param x:
        :param t:
        :return:
        """
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']

        batch_num = x.shape[0]

        # forward
        a1 = np.dot(x, W1) + b1
        z1 = self._sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = self._softmax(a2)

        # backward
        grads = {}
        dy = (y - t) / batch_num
        grads['W2'] = np.dot(z1.T, dy)
        grads['b2'] = np.sum(dy, axis=0)

        dz1 = np.dot(dy, W2.T)
        da1 = self.sigmoid_grad(a1) * dz1
        grads['W1'] = np.dot(x.T, da1)
        grads['b1'] = np.sum(da1, axis=0)

        return grads


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


def main2():

    # 画像サイズを定義(mnistを想定)
    image_size = 28*28

    # 2層ニューラルネットワークを生成する
    net = TwoLayerNet(input_size=image_size,    # 入力層のニューロン数
                      hidden_size=100,          # 隠れ層のニューロン数
                      output_size=10)           # 出力層のニューロン数

    # ダミーの入力データを作成（画像100枚分）
    x = np.random.rand(100, image_size)

    # 推論処理のテスト
    y = net.predict(x)

    # ダミーの正解ラベルを生成（画像100枚分）
    t = np.random.rand(100, 10)

    # 数値微分でxの勾配を求める
    # gradient = net.numerical_gradient(x, t)
    gradient = net.gradient(x, t)

    # result = net.accuracy(gradient, t)
    # print('result: ', result)
    print('gradient.keys: ', gradient.keys())

    loss_result = net.loss(x=x, t=t)
    print('loss result: ', loss_result)

    pass

if __name__ == '__main__':
    # main()
    main2()


