import numpy as np
from mnist import load_mnist    # mnistライブラリのload_mnist関数をインポート


def label_cross_entrory_error(y: np.ndarray, t: np.ndarray):
    """
    交差エントロピー誤差を返す

    :param y: ニューラルネットワークの出力
    :param t: ラベル表現の教師データ
    :return: 損失関数の計算結果
    """
    if y.ndim == 1:
        # ニューラルネットワークが1次元データ(n, )の場合、バッチ処理用に(1, n)へ整形する
        label_t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    batch_size = y.shape[0]

    # 微小の値: deltaを宣言
    # 1e-2: 1 x 10**-2 = 0.01
    # 1e-7: 1 x 10**-7 = 0.0000001
    delta = 1e-7

    # np.log(0)の時、マイナスの無限大-infになり、暴走してしまうので、deltaを加える
    result = -np.sum(np.log(y[np.arange(batch_size), t] + delta)) / batch_size
    return result


def main():

    # one-hot表現でMNISTデータセットを取得する
    (train_images, train_labels), (test_images, test_labels) = load_mnist(
        normalize=True,
        one_hot_label=False,
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

    # t: ラベル表現の教師データ
    one_hot_t = np.array([2, 8])

    # y: ニューラルネットワークの出力, 2の確率を最も高くする
    y = np.array([
        [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0, 0.1, 0, 0],
        [0.1, 0.05, 0.3, 0.0, 0.05, 0.1, 0, 0.0, 0.9, 0],
    ])

    result = label_cross_entrory_error(y, one_hot_t)

    print(result)


if __name__ == '__main__':
    main()
