import numpy as np
import matplotlib.pylab as plotlib

# def step_function(x):
#     if x > 1:
#         return 1
#     else:
#         return 0


def step_function(x: np.array):
    # 各要素 0以上か判定(bool)し、int型に変換した結果をリストにして返す
    return np.array(x > 0, dtype=int)


def main():
    # -5.0〜5.0まで0.1刻みの配列
    x = np.arange(-5.0, 5.0, 0.1)

    # xをステップ関数にかける
    y = step_function(x)

    plotlib.plot(x, y, label='step function result')
    plotlib.ylim(-0.1, 1.1)     # yの範囲を指定
    plotlib.xlabel('x')
    plotlib.ylabel('y')
    plotlib.legend()

    plotlib.show()


if __name__ == '__main__':
    main()
