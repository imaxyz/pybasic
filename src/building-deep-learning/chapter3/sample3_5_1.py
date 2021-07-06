import numpy as np


def softmax(a):
    exp_a = np.exp(a)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a

    return y


def softmax2(a):
    c = np.max(a)

    # オーバーフロー対策をしながら指数関数を計算する
    exp_a = np.exp(a - c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a

    return y


def main():

    a = np.array([1000.3, 2200.9, 2400.0])
    # y = softmax(a)
    y = softmax2(a)

    print(y)

if __name__ == '__main__':
    main()
