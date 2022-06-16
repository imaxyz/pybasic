import matplotlib.pyplot as plt
from optimizers import *


def f(x, y):
    return x ** 2 / 8.0 + y ** 2


def df(x, y):
    return x / 10.0, 2.0 * y


def main():
    """
    様々なパラメータの更新方法の特徴を可視化するサンプルプログラム
    """

    # パラメータの初期座標
    init_pos = (-7.0, 2.0)

    # 重みパラメータを初期化
    params = {'x': init_pos[0], 'y': init_pos[1]}

    # 勾配を初期化
    grads = {'x': 0, 'y': 0}

    # 各種最適化アルゴリズムを、学習率を与えつつ生成
    optimizers = OrderedDict()
    optimizers["SGD"] = SGD(lr=0.95)
    optimizers["Momentum"] = Momentum(lr=0.1)
    optimizers["AdaGrad"] = AdaGrad(lr=1.5)
    optimizers["Adam"] = Adam(lr=0.3)

    idx = 1

    # 各種最適化アルゴリズム
    for optimizer_name in optimizers:

        # 使用する最適化アルゴリズムを取得
        optimizer = optimizers[optimizer_name]

        # 軸ごとの履歴を格納するリストを初期化
        x_history = []
        y_history = []

        # 重みパラメータの初期値を取得
        params['x'], params['y'] = init_pos[0], init_pos[1]

        for i in range(30):

            # 各軸の履歴に、重みパラメータの推移を記録
            x_history.append(params['x'])
            y_history.append(params['y'])

            # パラメータの勾配を求める
            grads['x'], grads['y'] = df(params['x'], params['y'])

            # 使用する最適化アルゴリズムで重みパラメータの更新
            optimizer.update(params, grads)

        # x, y軸における値の範囲を指定
        x = np.arange(-10, 10, 0.01)
        y = np.arange(-5, 5, 0.01)

        X, Y = np.meshgrid(x, y)
        Z = f(X, Y)

        # for simple contour line
        mask = Z > 10
        Z[mask] = 0

        # plot
        # plt.subplot(4, 1, idx)    # 4行1列
        plt.subplot(2, 2, idx)  # 2行2列

        idx += 1

        # plt.plot(x_history, y_history, 'o-', color="red")
        plt.plot(x_history, y_history, 'ro-', linewidth=1, markersize=2)

        # 輪郭線をプロットする
        # plt.contour(X, Y, Z)
        plt.ylim(-10, 10)
        plt.xlim(-10, 10)
        plt.plot(0, 0, '+')
        plt.title(optimizer_name)
        plt.xlabel("x")
        plt.ylabel("y")

    plt.show()


if __name__ == '__main__':
    main()
