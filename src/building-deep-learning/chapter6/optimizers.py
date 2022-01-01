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

