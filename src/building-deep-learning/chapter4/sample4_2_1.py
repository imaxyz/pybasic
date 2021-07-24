import numpy as np


def sum_squared_error(y: np.ndarray, t: np.ndarray):
    """２乗和誤差を返す"""
    return 0.5 * np.sum((y-t)**2)


def main():

    # 2を正解とする
    t = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0])

    # 2の確率を最も高くする
    y = np.array([0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0, 0.1, 0, 0])
    y1 = sum_squared_error(y, t)

    # 7の確率を最も高くする
    y = np.array([0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0, 0.6, 0, 0])
    y2 = sum_squared_error(y, t)

    print(y1, y2)


if __name__ == '__main__':
    main()
