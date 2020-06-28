from __future__ import absolute_import, division, print_function, unicode_literals

# TensorFlow をインストール
import tensorflow as tf

# MNIST データセットをロードして準備します。サンプルを整数から浮動小数点数に変換します。
mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0


# 層を積み重ねてtf.keras.Sequentialモデルを構築します。訓練のためにオプティマイザと損失関数を選びます。
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# モデルを訓練してから評価します。
model.fit(x_train, y_train, epochs=5)
model.evaluate(x_test, y_test, verbose=2)
