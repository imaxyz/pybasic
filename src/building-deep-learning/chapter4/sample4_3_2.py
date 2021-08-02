import numpy as np
import matplotlib.pyplot as plt


def function_2(x):
    """今回定義する、サンプルの二次関数"""
    return 0.01*x**2 + 0.1*x


def numerical_diff(func, x):
    """数値微分で微分を求める"""
    h = 1e-4
    return (func(x+h) - func(x=x-h)) / (2*h)


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

    # 0-20まで0.1間隔のデータを作成する
    x = np.arange(0, 20, 0.1)

    # 対象の関数の結果リストを求める
    result = function_2(x)
    print('result: ', result)

    # 対象の関数のグラフを描画する
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.plot(x, result)

    # 指定した関数、指定した値における接線の関数（無名関数）を求める
    tangent_lambda = get_tangent_line_lambda(function_2, 12)
    result2 = tangent_lambda(x)

    # 接線をグラフに描画する
    plt.plot(x, result2)

    # グラフを表示する
    plt.show()


if __name__ == '__main__':
    main()
