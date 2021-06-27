import numpy as np
import matplotlib.pylab as plotlib


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def main():
    # -5.0〜5.0まで0.1刻みの配列
    x = np.arange(-5.0, 5.0, 0.1)

    # xをステップ関数にかける
    y = sigmoid(x)

    plotlib.plot(x, y, label='sigmoid function result')
    plotlib.ylim(-0.1, 1.1)     # yの範囲を指定
    plotlib.xlabel('x')
    plotlib.ylabel('y')
    plotlib.legend()

    plotlib.show()


if __name__ == '__main__':
    main()
