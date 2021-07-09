import sys, os
import numpy as np
from mnist import load_mnist    # mnistライブラリのload_mnist関数をインポート
from PIL import Image

sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定


def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()


def main():

    # (訓練画像、訓練ラベル), (テスト画像、テストラベル)の形式でmnistデータを取得
    (x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)

    img = x_train[0]
    label = t_train[0]
    print(label)  # 5

    print(img.shape)  # (784,)
    img = img.reshape(28, 28)  # 形状を元の画像サイズに変形
    # img = img.reshape(50, 50)  # 形状を元の画像サイズに変形
    print(img.shape)  # (28, 28)

    img_show(img)
    print('hello, mnist!')


if __name__ == '__main__':
    main()
