import numpy as np


def cross_entropy_error(y: np.ndarray, t: np.ndarray):
    """交差エントロピー誤差を返す"""

    # 微小の値: deltaを宣言
    # 1e-2: 1 x 10**-2 = 0.01
    # 1e-7: 1 x 10**-7 = 0.0000001
    delta = 1e-7

    # np.log(0)の時、マイナスの無限大-infになり、暴走してしまうので、deltaを加える
    result = -np.sum(t * np.log(y + delta))
    return result


def main():

    # 2を正解とする
    t = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0])

    # 2の確率を最も高くする
    y = np.array([0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0, 0.1, 0, 0])
    y1 = cross_entropy_error(y, t)

    # 7の確率を最も高くする
    y = np.array([0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0, 0.6, 0, 0])
    y2 = cross_entropy_error(y, t)

    print(y1, y2)


if __name__ == '__main__':
    main()
