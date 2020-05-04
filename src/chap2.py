filename = 'sample.txt'

with open(filename, 'wt', encoding='utf-8') as fp:
    fp.write('This is sample.')

# fp = open(filename, 'rt')
# fp.read()   # ファイル内容を全部読み込み
# fp.readline()
