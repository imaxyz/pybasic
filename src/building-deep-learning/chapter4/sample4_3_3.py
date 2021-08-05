import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D     # 3Dグラフを描画するパッケージ


def function_3(x0, x1):
    """f(x0, x1) = x0**2 + x1**2"""
    return x0**2 + x1**2


def function_tmp1(x0):
    """偏微分のための、x1=4で固定した新しい関数"""
    return x0**2 + 4.0**2


def function_tmp2(x1):
    """偏微分のための、x0=3で固定した新しい関数"""
    return 3.0**2 + x1**2


def function_3b(x):
    """f(x0, x1) = x0**2 + x1**2"""
    return np.sum(x**2)


def numerical_diff(func, x):
    """数値微分で微分を求める"""
    h = 1e-4
    return (func(x+h) - func(x-h)) / (2*h)


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
    intercept = func(x) - d*x
    print('切片: ', intercept)

    # 接線の方程式を作成して返す
    return lambda t: d*t + intercept


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

    x0_result = numerical_diff(func=function_tmp1, x=3.0)
    x1_result = numerical_diff(func=function_tmp2, x=4.0)

    print('x0_result: ', x0_result)
    print('x1_result: ', x1_result)


if __name__ == '__main__':
    # main()
    main2()
