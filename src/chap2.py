import sys
# python のバージョンを出力する
print(sys.version)

filename = 'sample.txt'

with open(filename, 'wt', encoding='utf-8') as fp:
    fp.write('This is sample.')
    # fp = open(filename, 'rt')
    # fp.read()
    # fp.readline()

for n in range(10):
    print('10進数: {0:d}, '
          '2進数(最低4桁): {0:04b}, '
          '2進数{0:0b}, '
          '8進数:{0:0o}, '
          '16進数:{0:0x}'.format(n))

# Pythonにおける浮動小数点の精度の問題

