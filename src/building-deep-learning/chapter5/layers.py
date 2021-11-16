import numpy as np


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
