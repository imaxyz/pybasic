import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 3Dグラフを描画するパッケージ


def function_3(x0, x1):
    """f(x0, x1) = x0**2 + x1**2"""
    return x0 ** 2 + x1 ** 2


def function_tmp1(x0):
    """偏微分のための、x1=4で固定した新しい関数"""
    return x0 ** 2 + 4.0 ** 2


def function_tmp2(x1):
    """偏微分のための、x0=3で固定した新しい関数"""
    return 3.0 ** 2 + x1 ** 2


def function_3b(x):
    """f(x0, x1) = x0**2 + x1**2"""
    return np.sum(x ** 2)
    # return x[0]**2 + x[1]**2


def _numerical_gradient_no_batch(func, x: np.ndarray):
    """
    xで指定された、ある点に関する各変数の勾配を求めて返す。
    （実質、引数の各要素ごとに数値微分を行なっているのみ。）

    :param func: 複数の変数を持つ関数
    :param x: np.ndarray(float64)
    :return: xの勾配
    """

    # 十分小さい値を定義する
    h = 1e-4  # 0.0001

    # 勾配を初期化する。(xと同じ形状で、その要素が全て0の配列とする)
    gradient = np.zeros_like(x)

    # x0, x1, ...ごとに偏微分を求める
    for idx in range(x.size):
        #
        tmp_val = x[idx]

        # ---- idxの値の数値微分をもとめる一連の処理
        x[idx] = float(tmp_val) + h
        result_x_plus_h = func(x)  # f(x+h)

        x[idx] = tmp_val - h
        result_x_minus_h = func(x)  # f(x-h)

        # 数値微分で微分を求めて、勾配として結果の配列に格納する
        gradient[idx] = (result_x_plus_h - result_x_minus_h) / (2 * h)

        # x[idx] = tmp_val  # 値を元に戻す

    return gradient


def numerical_diff(func, x):
    """数値微分で微分を求める"""
    h = 1e-4

    result_x_plus_h = func(x + h)  # f(x+h)
    result_x_minus_h = func(x - h)  # f(x-h)

    return (result_x_plus_h - result_x_minus_h) / (2 * h)


def get_tangent_line_lambda(func, x):
    """
    二次関数接線の方程式を返す
    :param func: 二次関数
    :param x: パラメータx
    :return: funcの接線の方程式（関数）
    """

    # xの微分(接線の傾き)を求める
    d = numerical_diff(func, x)
    print('微分: ', d)

    # 切片を求める
    # 直線の式 y = a*x + b ---> func(x) = d*x + intercept
    intercept = func(x) - d * x
    print('切片: ', intercept)

    # 接線の方程式を作成して返す
    return lambda t: d * t + intercept


def main():
    # 変数を定義
    x0 = np.arange(-3.0, 3.0, 0.25)
    x1 = np.arange(-3.0, 3.0, 0.25)

    # x0, x1から格子列を作成する
    X0, X1 = np.meshgrid(x0, x1)
    result = function_3(X0, X1)

    # 3Dの図を作成
    figure = plt.figure()
    axes_3d = Axes3D(figure)
    axes_3d.set_xlabel("x0")
    axes_3d.set_ylabel("x1")
    axes_3d.set_zlabel("f(x)")

    # データをワイヤーフレームで描画
    axes_3d.plot_wireframe(X0, X1, result)

    # グラフを表示する
    plt.show()


def main2():
    # 1つの変数ごとに偏微分を求める

    # x0の偏微分のため、x1の値を固定した関数でx0の微分を求める
    x0_result = numerical_diff(func=function_tmp1, x=3.0)

    # x1の偏微分のため、x0の値を固定した関数でx1の微分を求める
    x1_result = numerical_diff(func=function_tmp2, x=4.0)

    print('x0_result: ', x0_result)
    print('x1_result: ', x1_result)


def main3():

    # [3, 4] で指定すると、ndarrayのdtypeがint64（整数型）になり、微細な値を扱えなくなるのでfloatで指定
    result1 = _numerical_gradient_no_batch(function_3b, np.array([3.0, 4.0]))
    print(result1)

    result2 = _numerical_gradient_no_batch(function_3b, np.array([0.0, 2.0]))
    print(result2)

    result3 = _numerical_gradient_no_batch(function_3b, np.array([3.0, 0.0]))
    print(result3)


if __name__ == '__main__':
    # main()
    # main2()
    main3()
