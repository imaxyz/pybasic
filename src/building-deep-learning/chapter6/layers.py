import sys, os
import numpy as np
from collections import OrderedDict

def numerical_gradient(f, x):
    """多次元配列に対応した勾配を求める"""
    h = 1e-4  # 0.0001
    grad = np.zeros_like(x)

    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])

    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
        x[idx] = tmp_val + h
        fxh1 = f(x)  # f(x+h)

        x[idx] = tmp_val - h
        fxh2 = f(x)  # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2 * h)

        x[idx] = tmp_val  # 値を元に戻す
        it.iternext()

    return grad


class MulLayer:
    """乗算レイヤー"""
    def __init__(self):
        self.x = None
        self.y = None

    def forward(self, x, y):
        self.x = x
        self.y = y
        out = x * y

        return out

    def backward(self, dout):
        dx = dout * self.y
        dy = dout * self.x

        return dx, dy


class ReLU:
    """Reluノード"""

    def __init__(self):
        # mask: インスタンス変数。
        #   True/FalseからなるNumpy配列
        self.is_minus_mask_array = None

    def forward(self, x: np.ndarray):
        """順伝播を実行する"""

        # 入力xが、0以下である場合True,
        #   0より大きい正の数の場合、False
        self.is_minus_mask_array = (x <= 0)

        # 入力値の内、マイナスの値は0に丸める
        out = x.copy()
        out[self.is_minus_mask_array] = 0

        # 入力値の内、マイナスの値を0に変換して返す
        return out

    def backward(self, dout):
        """逆伝播を実行する"""

        # 順伝播の入力がマイナスだった場合、
        #   逆伝播では0として下流への信号はそこで止める。
        dout[self.is_minus_mask_array] = 0

        # 変換済みの微分値を返す
        return dout


class SigmoidLayer:
    """Sigmoidレイヤ"""
    def __init__(self):
        # 順伝搬時の出力
        self.forward_out = None

    def forward(self, x):
        """順伝播の計算"""

        # Sigmoid関数の計算: y = sigmoid(x)
        out = (1 / (1 + np.exp(-x)))

        # 順伝播時に出力を保持しておく
        self.forward_out = out

        return out

    def backward(self, dout):
        """逆伝播の計算"""

        # 入力された微分に y(1-y)を乗算する
        dx = dout * self.forward_out * (1.0 - self.forward_out)

        # dx: 下流に渡す値
        return dx


class Affine:
    """バッチ対応版Affineレイヤ"""

    def __init__(self, W, b):
        self.W = W  # 重み
        self.b = b  # バイアス

        self.x = None   # 入力
        self.original_x_shape = None

        # 重み・バイアスパラメータの微分
        self.dW = None
        self.db = None

    def forward(self, x):
        """順伝播"""

        # テンソル(4次元データ)対応
        self.original_x_shape = x.shape
        x = x.reshape(x.shape[0], -1)

        # 入力値を保持
        self.x = x

        # 順伝播のAffine変換
        out = np.dot(self.x, self.W) + self.b

        return out

    def backward(self, dout):
        """逆伝播"""

        # dx: 渡されたdoutと、Wの転置の行列積を求める ... 公式より
        dx = np.dot(dout, self.W.T)

        # dw: 入力xの転置と、渡されたdoutの行列積を求める
        self.dW = np.dot(self.x.T, dout)

        # バイアスの逆伝播
        self.db = np.sum(dout, axis=0)

        dx = dx.reshape(*self.original_x_shape)  # 入力データの形状に戻す（テンソル対応）
        return dx


class SoftmaxWithLoss:
    """
    Softmax-with-Lossレイヤ。
    Softmax関数と、交差エントロピー誤差を含む。
    """

    def __init__(self):
        self.loss = None
        self.y = None  # softmaxの出力
        self.t = None  # 教師データ

    def _softmax(self, x):
        """Softmax関数 (3.5.2章を参照)"""
        x = x - np.max(x, axis=-1, keepdims=True)  # オーバーフロー対策
        return np.exp(x) / np.sum(np.exp(x), axis=-1, keepdims=True)

    def _cross_entropy_error(self, y, t):
        """クロスエントロピー誤差。(4.2.4章を参照)"""
        if y.ndim == 1:
            t = t.reshape(1, t.size)
            y = y.reshape(1, y.size)

        # 教師データがone-hot-vectorの場合、正解ラベルのインデックスに変換
        if t.size == y.size:
            t = t.argmax(axis=1)

        batch_size = y.shape[0]
        return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size

    def forward(self, x, t):
        """順伝播"""

        # 教師データを取得
        self.t = t

        # ソフトマックス関数の出力を取得
        self.y = self._softmax(x)

        # 損失を取得
        self.loss = self._cross_entropy_error(self.y, self.t)

        # 損失を返す
        return self.loss

    def backward(self, dout=1):
        """逆伝播"""

        # バッチサイズを取得
        batch_size = self.t.shape[0]

        if self.t.size == self.y.size:  # 教師データがone-hot-vectorの場合
            # 逆伝搬の計算
            # 伝播する値を、バッチ個数で割ることで、平均化した誤差を前レイヤに伝播する
            dx = (self.y - self.t) / batch_size
        else:
            # 逆伝搬の計算
            dx = self.y.copy()

            # TODO: この計算の意味を確認
            dx[np.arange(batch_size), self.t] -= 1

            # 伝播する値を、バッチ個数で割ることで、平均化した誤差を前レイヤに伝播する
            dx = dx / batch_size

        return dx


class TwoLayerNet:
    """
    2層ニューラルネットワーク
    既存のレイヤを再利用することによって、複雑な処理をレイヤの伝播のみで実現できる。
    """

    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        """
        初期化処理

        :param input_size: 入力層のニューロン数
        :param hidden_size: 隠れ層のニューロン数
        :param output_size: 出力層のニューロン数
        :param weight_init_std: 重み初期化時におけるガウス分布のスケール
        """
        # 重みの初期化

        # params: ニューラルネットワークの各種パラメータを格納する辞書
        self.params = {}

        # 第1層の重みを初期化
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)

        # 第1層のバイアスを初期化
        self.params['b1'] = np.zeros(hidden_size)

        # 第2層の重みを初期化
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)

        # 第2層のバイアスを初期化
        self.params['b2'] = np.zeros(output_size)

        # ニューラルネットワークのレイヤを保持する順序付き辞書を生成
        self.layers = OrderedDict()

        # 順序を意識して各種レイヤーを追加していく
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['Relu1'] = ReLU()
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])

        self.lastLayer = SoftmaxWithLoss()

    def _predict(self, x):
        """認識結果（各レイヤーを順伝播した結果）を得る"""

        # 各レイヤーの順伝播を実行する
        for layer in self.layers.values():
            x = layer.forward(x)

        return x

    def loss(self, x, t):
        """
        損失関数の値を求める

        x: 入力（画像データ）
        t: 教師データ（正解ラベル）
        """

        # 各レイヤーを順伝播した結果を取得
        y = self._predict(x)

        # 最後のレイヤーの順伝播を実行した結果を返す
        result = self.lastLayer.forward(y, t)

        return result

    def accuracy(self, x, t):
        """認識精度を求める"""
        y = self._predict(x)
        y = np.argmax(y, axis=1)

        if t.ndim != 1:
            t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    def numerical_gradient(self, x, t):
        """
        重みパラメータに対する勾配を、数値微分によって求める
        x: 入力（画像データ）
        t: 教師データ（正解ラベル）
        """
        loss_W = lambda W: self.loss(x, t)

        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])

        return grads

    def gradient(self, x, t):
        """
        重みパラメータに対する勾配を、誤差逆伝播法により求める
        x: 入力（画像データ）
        t: 教師データ（正解ラベル）
        """
        # 損失関数の値を求める（最初から最後までのレイヤーを順伝播した結果）
        self.loss(x, t)

        # 最後尾のレイヤーを逆伝播した値を取得する
        dout = 1
        dout = self.lastLayer.backward(dout)

        # 最後尾のレイヤーから、誤差逆伝播によって、重みパラメータに対する
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # 重みパラメータに対する勾配を、返却値に設定する
        grads = {}
        grads['W1'], grads['b1'] = self.layers['Affine1'].dW, self.layers['Affine1'].db
        grads['W2'], grads['b2'] = self.layers['Affine2'].dW, self.layers['Affine2'].db

        return grads