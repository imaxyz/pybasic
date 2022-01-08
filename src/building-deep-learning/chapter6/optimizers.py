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


class Momentum:
    """Momentum"""

    def __init__(self, lr=0.01, momentum=0.9):
        """

        :param lr: 学習係数
        :param momentum:
        """
        self.lr = lr
        self.momentum = momentum

        # 物体の速度
        self.v = None

    def update(self, params, grads):
        """

        :param params:
        :param grads:
        :return:
        """

        # 速度vが無い時は、paramsと同じ構造をゼロで初期化した内容でvを初期化する
        if self.v is None:
            self.v = {}
            for key, val in params.items():
                # np.zeros_like(): valと同じ構造を持ち、値がゼロのディクショナリを生成する
                self.v[key] = np.zeros_like(val)

        for key in params.keys():

            # Momentumの計算。vを更新する
            self.v[key] = self.momentum * self.v[key] - self.lr * grads[key]

            # 重みパラメータを更新。重みパラメータにvを加算する
            params[key] += self.v[key]


class AdaGrad:
    """AdaGrad"""

    def __init__(self, lr=0.01):
        """

        :param lr:
        """
        self.lr = lr
        self.h = None

    def update(self, params, grads):
        """

        :param params:
        :param grads: 重みWに関する損失関数の勾配
        :return:
        """

        # 速度vが無い時は、paramsと同じ構造をゼロで初期化した内容でhを初期化する
        if self.h is None:
            self.h = {}
            for key, val in params.items():
                self.h[key] = np.zeros_like(val)

        for key in params.keys():
            # 損失関数の勾配の二乗和をhに加算する
            self.h[key] += grads[key] * grads[key]

            # パラメータを更新する。
            # パラメータ毎にhの平方根で割って、学習のスケールを調整し、学習係数の減衰を行う
            params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key]) + 1e-7)


class Adam:
    """
    Adam (http://arxiv.org/abs/1412.6980v8)
    MomentumとAdaGradのアイディアを参考に設計された最適化アルゴリズム
    """

    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999):
        """
        ３つのハイパーパラメータを取るイニシャライザ
        :param lr:
        :param beta1:
        :param beta2:
        """
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.iter = 0
        self.m = None
        self.v = None

    def update(self, params, grads):
        if self.m is None:
            self.m, self.v = {}, {}
            for key, val in params.items():
                self.m[key] = np.zeros_like(val)
                self.v[key] = np.zeros_like(val)

        self.iter += 1
        lr_t = self.lr * np.sqrt(1.0 - self.beta2 ** self.iter) / (1.0 - self.beta1 ** self.iter)

        for key in params.keys():
            self.m[key] += (1 - self.beta1) * (grads[key] - self.m[key])
            self.v[key] += (1 - self.beta2) * (grads[key] ** 2 - self.v[key])

            params[key] -= lr_t * self.m[key] / (np.sqrt(self.v[key]) + 1e-7)