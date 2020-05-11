import numpy as np
import requests
from Crypto.PublicKey import RSA

# numpyで二次元配列を生成してみる
a = np.array([[1, 2], [3, 4]])
print(a)

# dir()を用いて、インポートしたモジュールで利用できる関数やクラスを列挙する
modules = dir(requests)
print(modules)

# 通信してみる
url = 'https://www.google.co.jp'
response = requests.get(url)
print(response.status_code, response.content)

# RSAの秘密鍵と公開鍵を作成してみる
rsa = RSA.generate(2048)
private_key = rsa.exportKey()
public_key = rsa.publickey()

# TODO: テキストで公開鍵を出力する
print(public_key)