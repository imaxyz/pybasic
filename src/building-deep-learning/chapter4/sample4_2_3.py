import numpy as np
from mnist import load_mnist    # mnistライブラリのload_mnist関数をインポート


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

    # one-hot表現でMNISTデータセットを取得する
    (train_images, train_labels), (test_images, test_labels) = load_mnist(
        normalize=True,
        one_hot_label=True,
    )

    # 訓練データのサイズを求める
    train_size = train_images.shape[0]

    # ミニバッチの規模を決める
    mini_batch_size = 10

    # ミニバッチとして選び出すインデックスを作成
    random_choice_indexes = np.random.choice(train_size, mini_batch_size)

    # ミニバッチのインデックスを用いて、全データの中から、所定のバッチサイズ分だけ無作為に訓練データを抽出する
    batch_train_images = train_images[random_choice_indexes]
    batch_train_labels = train_labels[random_choice_indexes]

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
