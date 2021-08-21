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


def gradient_descent(func, init_x, learning_rate=0.01, step_num=100):
    """
    勾配降下法で学習する

    :param func: 対象の関数
    :param init_x: 初期値
    :param learning_rate: 学習率
    :param step_num: 勾配法による繰り返しの数

    :return: 処理結果, 処理結果の履歴
    """

    x = init_x
    logs = [x.copy()]

    for i in range(step_num):
        x -= learning_rate * single_numerical_gradient(func, x)
        logs.append(x.copy())

    return x, np.array(logs)


def single_numerical_gradient(func, x: np.ndarray):
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

        # 値を元に戻す。この処理が無いと、勾配降下法で結果が異常になる。
        x[idx] = tmp_val

    return gradient


def get_numerical_gradient(f, matrix: np.ndarray):
    """勾配を求めて返す"""
    if matrix.ndim == 1:
        return single_numerical_gradient(f, matrix)
    else:
        grad = np.zeros_like(matrix)

        for idx, x in enumerate(matrix):
            # (e.g) x: [-1.5001 -2.0001]
            grad[idx] = single_numerical_gradient(f, x)

        return grad


def get_numerical_diff(func, x):
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
    d = get_numerical_diff(func, x)
    print('微分: ', d)

    # 切片を求める
    # 直線の式 y = a*x + b ---> func(x) = d*x + intercept
    intercept = func(x) - d * x
    print('切片: ', intercept)

    # 接線の方程式を作成して返す
    return lambda t: d * t + intercept


def main():
    # -3.0〜3.0までの値をもつリストを定義
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
    x0_result = get_numerical_diff(func=function_tmp1, x=3.0)

    # x1の偏微分のため、x0の値を固定した関数でx1の微分を求める
    x1_result = get_numerical_diff(func=function_tmp2, x=4.0)

    print('x0_result: ', x0_result)
    print('x1_result: ', x1_result)


def main3():
    # 各点における勾配を求める

    # [3, 4] で指定すると、ndarrayのdtypeがint64（整数型）になり、微細な値を扱えなくなるのでfloatで指定
    result1 = single_numerical_gradient(function_3b, np.array([3.0, 4.0]))
    print(result1)

    result2 = single_numerical_gradient(function_3b, np.array([0.0, 2.0]))
    print(result2)

    result3 = single_numerical_gradient(function_3b, np.array([3.0, 0.0]))
    print(result3)


def main4():
    # 勾配を矢印で図に描画する

    x0 = np.arange(-5, 5, 0.25)
    x1 = np.arange(-5, 5, 0.25)
    X, Y = np.meshgrid(x0, x1)

    X = X.flatten()
    Y = Y.flatten()

    # (2, 1600)行列の作成
    matrix_orig = np.array([X, Y])

    # (1600, 2)行列の作成(転置行列)
    matrix = matrix_orig.T

    # (1600, 2)行列の勾配を求めて、(2, 1600)の形に戻す。(再び、転置行列を得る）
    # 2... (X [...], Y[...])
    gradient = get_numerical_gradient(function_3b, matrix).T

    # 新しい図を生成する
    plt.figure()

    # 矢印をプロット（グラフを描画）する
    plt.quiver(X, Y,
               # 勾配ベクトルの結果にマイナスをつけて、方向を逆向きにする
               -gradient[0],    # (1600, )
               -gradient[1],    # (1600, )
               angles="xy", color="#3333ff")
    # plt.quiver(X, Y, grad[0], grad[1],  angles="xy", color="#ff6666")

    # グラフの範囲を-5.5〜5とする
    plt.xlim([-5.5, 5])
    plt.ylim([-5.5, 5])

    plt.xlabel('x0')
    plt.ylabel('x1')

    # グラフに、グリッドを描画する
    plt.grid()

    # 図を再描画する
    # plt.draw()

    # 図を表示する
    plt.show()


def main5():
    # 勾配降下法で最小値を探索する
    result, logs = gradient_descent(function_3b,
                                    init_x=np.array([-3.0, 4.0]),
                                    learning_rate=0.1, step_num=100)
    print('result: ', result)


def main6():

    # 勾配降下法で最小値を探索する
    result, logs = gradient_descent(function_3b,
                                    init_x=np.array([-3.0, 4.0]),
                                    learning_rate=0.1, step_num=100)

    # 点線を打つ[xデータ], [yデータ], フォーマット文字列
    # フォーマット文字列の仕様: https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.plot.html#matplotlib.pyplot.plot
    # b- : 青の実線
    # ro : 赤の円
    # plt.plot([-5, 5], [0, 0], 'b--')    # 横線
    # plt.plot([-5, 5], [0, 0], 'b-.')    # 横線
    plt.plot([-5, 5], [0, 0], 'k:')  # 横線
    # plt.plot([-4, 3], [0, 0], '^g')    # 横線
    # plt.plot([0, 0], [-5, 5], '--b')  # 縦線
    plt.plot([0, 0], [-5, 5], ':k')  # 縦線

    # plt.plot(x_history[:, 0], x_history[:, 1], 'o')
    plt.plot(logs[:, 0], logs[:, 1], '^g')

    # 関数の最小値をプロット
    plt.plot(result[0], result[1], 'ro')

    plt.xlim(-3.5, 3.5)
    plt.ylim(-4.5, 4.5)
    plt.xlabel("X0")
    plt.ylabel("X1")
    plt.show()


def main7():

    # 学習率が大き過ぎる例
    result1, logs1 = gradient_descent(function_3b,
                                    init_x=np.array([-3.0, 4.0]),
                                    learning_rate=10.0, step_num=100)

    # 学習率が小さ過ぎる例
    result2, logs2 = gradient_descent(function_3b,
                                    init_x=np.array([-3.0, 4.0]),
                                    learning_rate=1e-10, step_num=100)

    # 点線を打つ[xデータ], [yデータ], フォーマット文字列
    # フォーマット文字列の仕様: https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.plot.html#matplotlib.pyplot.plot
    # b- : 青の実線
    # ro : 赤の円
    # plt.plot([-5, 5], [0, 0], 'b--')    # 横線
    # plt.plot([-5, 5], [0, 0], 'b-.')    # 横線
    plt.plot([-5, 5], [0, 0], 'k:')  # 横線
    # plt.plot([-4, 3], [0, 0], '^g')    # 横線
    # plt.plot([0, 0], [-5, 5], '--b')  # 縦線
    plt.plot([0, 0], [-5, 5], ':k')  # 縦線

    # plt.plot(x_history[:, 0], x_history[:, 1], 'o')
    plt.plot(logs1[:, 0], logs1[:, 1], '^b')
    plt.plot(logs2[:, 0], logs2[:, 1], '^r')

    # 関数の最小値をプロット
    plt.plot(result1[0], result1[1], 'bo')
    plt.plot(result2[0], result2[1], 'go')

    plt.xlim(-3.5, 3.5)
    plt.ylim(-4.5, 4.5)
    plt.xlabel("X0")
    plt.ylabel("X1")
    plt.show()
    pass

if __name__ == '__main__':
    # main()
    # main2()
    # main3()
    # main4()
    # main5()
    # main6()
    main7()
