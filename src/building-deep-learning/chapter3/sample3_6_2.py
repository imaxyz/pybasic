import sys, os
import numpy as np
from mnist import load_mnist    # mnistライブラリのload_mnist関数をインポート
from PIL import Image


def get_normalized_test_data():
    """正規化されたテストデータを返す"""

    (train_images, train_labels), (test_images, test_labels) = load_mnist(
        normalize=True,
        flatten=True,
        one_hot_label=False,
    )

    return test_images, test_labels


def init_network():

    pass


def main():

    sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定

    # (訓練画像、訓練ラベル), (テスト画像、テストラベル)の形式でmnistデータを取得
    (train_images, train_labels), (test_images, test_labels) = load_mnist(
        # 画像を一次元配列にするか
        flatten=True,

        # 画像のピクセル値を0.0~1.0に正規化するか
        normalize=False,

        # one-hot表現を行うか
        one_hot_label=False,
    )

    # 訓練データの最初の要素を取得
    train_image = train_images[0]
    train_label = train_labels[0]

    print('train_image.shape: ', train_image.shape)  # (784,)
    print('train_label: ', train_label)  # 5

    # 形状を元の画像サイズに変形
    # MNISTの画像データのサイズは、28x28
    train_image = train_image.reshape(28, 28)
    # img = img.reshape(50, 50)  # 形状を元の画像サイズに変形
    print('train_image reshaped: ', train_image.shape)  # (28, 28)



if __name__ == '__main__':
    main()
