import numpy as np


def AND(x1, x2):
    """
    重みとバイアスを用いた方式で実装するANDゲート (AND論理回路)
    重みつき入力の総和が閾値シータを超えると1を返す。それ以外は0を返す。

    :param x1: 入力1
    :param x2: 入力2
    :return: 結果
    """
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.7
    tmp = np.sum(w*x) + b

    if tmp <= 0:
        return 0
    else:
        return 1


def NAND(x1, x2):
    """
    重みとバイアスを用いた方式で実装するNANDゲート (NAND論理回路)
    重みつき入力の総和が閾値シータを超えると1を返す。それ以外は0を返す。

    :param x1: 入力1
    :param x2: 入力2
    :return: 結果
    """
    x = np.array([x1, x2])
    w = np.array([-0.5, -0.5])
    b = 0.7
    tmp = np.sum(w*x) + b

    if tmp <= 0:
        return 0
    else:
        return 1


def OR(x1, x2):
    """
    重みとバイアスを用いた方式で実装するORゲート (OR論理回路)
    重みつき入力の総和が閾値シータを超えると1を返す。それ以外は0を返す。

    :param x1: 入力1
    :param x2: 入力2
    :return: 結果
    """
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.2
    tmp = np.sum(w*x) + b

    if tmp <= 0:
        return 0
    else:
        return 1


def main():
    print('---AND---')
    print(f'(0, 0): {AND(0, 0)}')
    print(f'(1, 0): {AND(1, 0)}')
    print(f'(0, 1): {AND(0, 1)}')
    print(f'(1, 1): {AND(1, 1)}')

    print('---NAND---')
    print(f'(0, 0): {NAND(0, 0)}')
    print(f'(1, 0): {NAND(1, 0)}')
    print(f'(0, 1): {NAND(0, 1)}')
    print(f'(1, 1): {NAND(1, 1)}')

    print('---OR---')
    print(f'(0, 0): {OR(0, 0)}')
    print(f'(1, 0): {OR(1, 0)}')
    print(f'(0, 1): {OR(0, 1)}')
    print(f'(1, 1): {OR(1, 1)}')


if __name__ == '__main__':
    main()
