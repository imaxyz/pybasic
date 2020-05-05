import sys
# python のバージョンを出力する
print(sys.version)

filename = 'src/sample.txt'

with open(filename, 'rt', encoding='utf-8') as fp:
    # fp.write('This is sample.')
    # fp = open(filename, 'rt')
    fp.read()
    # fp.readline()

for n in range(10):
    print('10進数: {0:d}, '
          '2進数(最低4桁): {0:04b}, '
          '2進数{0:0b}, '
          '8進数:{0:0o}, '
          '16進数:{0:0x}'.format(n))

# Pythonにおける浮動小数点の精度の問題
print(1/3)
print(7/3)

# print()の演習
print('left', 'right', sep='|')

for i in range(5):
    print(i, end=' ')