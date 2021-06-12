def AND(x1, x2):
    """
    ANDゲート (AND論理回路)
    重みつき入力の総和が閾値シータを超えると1を返す。それ以外は0を返す。

    :param x1: 入力1
    :param x2: 入力2
    :return: 結果
    """
    w1, w2, theta = 0.5, 0.5, 0.7
    tmp = x1*w1 + x2*w2

    if tmp <= theta:
        return 0
    elif tmp > theta:
        return 1


def main():
    print(AND(0, 0))
    print(AND(1, 0))
    print(AND(0, 1))
    print(AND(1, 1))


if __name__ == '__main__':
    main()
