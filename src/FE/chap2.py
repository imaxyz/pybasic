import sys
# python のバージョンを出力する
print(sys.version)

filename = 'data/sample.txt'

try:
    with open(filename, 'rt', encoding='utf-8') as fp:
        # fp.write('This is sample.')
        # fp = open(filename, 'rt')
        fp.read()
        # fp.readline()
        pass
except Exception as e:
    print(e)

for n in range(10):
    print('10進数: {0:d}, '
          '2進数(最低4桁): {0:04b}, '
          '2進数{0:0b}, '
          '8進数:{0:0o}, '
          '16進数:{0:0x}'.format(n))

# Pythonにおける浮動小数点の精度の問題の確認
print(1 / 3)
print(7 / 3)

# print()の演習
print('left', 'right', sep='|')

for i in range(5):
    # print()の後に挿入する文字を、スペースに変更する。
    print(i, end=' ')

# ファイルを開いて、ファイルにprint()で出力する
with open('../../data/chp2-2.txt', 'at') as fp:
    print('Hello, world!', file=fp)

# 例外の補足
try:
    filename2 = 'wrong-hoge.txt'
    with open(filename2, 'rt', encoding='utf-8') as fp:
        any_str = fp.read()
        print('any: {0}'.format(any_str))

except OSError as e:
    print(e)
except (ValueError, ZeroDivisionError, TypeError) as e:
    print(e)
except Exception as e:
    print(e)
finally:
    print('finally: {0}'.format(filename2))
    pass

# ユーザが発生させる例外
try:
    raise ValueError('ユーザーが発生させたValueError')
except ValueError as e:
    print(e, 'という例外が発生した。')


# ユーザー定義例外
class SomeAnonymousError(Exception):
    pass


try:
    raise SomeAnonymousError('ユーザー定義の例外')
except SomeAnonymousError as e:
    print(e, 'という例外が発生した')

# 例外処理の挙動確認
print('実行結果: ', end=' ')
try:
    raise NameError('名前エラー')
except Exception as e:  # (!) NameErrorよりも先に補足してしまう
    print('例外が発生しました: ', e)
except NameError as e:
    print('NameErrorが発生しました', e)
finally:
    print('終わり')
