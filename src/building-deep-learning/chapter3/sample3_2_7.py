import numpy as np
import matplotlib.pylab as plotlib


def relu(x):
    return np.maximum(0, x)


def main():
    # -5.0〜5.0まで0.1刻みの配列
    x = np.arange(-5.0, 5.0, 0.1)

    # xをReLU関数にかける
    y = relu(x)

    plotlib.plot(x, y, label='relu() result')
    plotlib.ylim(-0.1, 5.1)     # yの範囲を指定
    plotlib.xlabel('x')
    plotlib.ylabel('y')
    plotlib.legend()

    plotlib.show()


if __name__ == '__main__':
    main()
