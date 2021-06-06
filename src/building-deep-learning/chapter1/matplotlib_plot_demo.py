# データを生成するために、NumPyをインポート
import numpy as np

# pyplotモジュールをインポート
import matplotlib.pyplot as plot

# データの作成

# NumPyのarange()メソッドで、0〜6まで、0.1刻みのデータを生成する。
x = np.arange(0, 6, 0.1)

# xのsin()を求める
y1 = np.sin(x)
y2 = np.cos(x)

# xとyのグラフを作成する
plot.plot(x, y1, label='sin(y1)')
plot.plot(x, y2, linestyle='--', label='cos(y2)')
plot.xlabel('x values')
plot.ylabel('y values')
plot.legend()   # 左下に実線の意味するラベルを表示する
plot.title('sin and cos values')

# グラフを表示する
plot.show()
