def sum_elements(digitlist: list):
    """
    パラメータの数字要素を合算して返す

    :param digitlist: 数字を含むリスト
    :return:
    """
    sum_digit = 0
    for digit in digitlist:
        if digit.isdigit():
            sum_digit += int(digit)

    return sum_digit


def execute_basic_python_functons():
    # 数字要素を合算する
    digitlist = ['1', '4', 'abc']
    sum_digit = sum_elements(digitlist)
    print(sum_digit)

    # input()のテスト
    # input_buffer = input('何か数字を入力してください: ')
    # if not input_buffer.isdigit():
    #     raise ValueError('数字じゃないよ', input_buffer)
    #
    # input_buffer = int(input_buffer)
    input_buffer = 2 ^ 8  # ^: 排他的論理和
    print(input_buffer,
          bin(input_buffer),
          oct(input_buffer),
          hex(input_buffer),
          abs(input_buffer), sep=', ')

    # べき乗を計算する
    pow_result = pow(2, -8)
    print('pow: ', pow_result)

    # 各種データ型変換
    int_value = int('5')
    float_value = float('5.2')
    str_value = str('5')

    tuple_sample = (0, 1, 2, 1, 2)
    listed = list(tuple_sample)

    list_sample = [0, 1, 2, 1, 2]
    tupled = tuple(list_sample)

    print(list(enumerate(list_sample)))

    # set()によって集合にすると、リストやタプルの重複要素が削除される
    setted1 = set(tuple_sample)
    setted2 = set(list_sample)

    # イテレータに演算を行う関数
    # enumerate()は、要素番号と要素の内容を返す
    print(list(enumerate(setted1)))
    print(list(enumerate(setted2)))

    for i, value in enumerate(digitlist):
        print('hoge: ', i, value, sep='__')

    # map()
    numlist = [1, 2, 4]

    def calc_double(x):
        return x * 2

    # map()の戻り値は、そのままでは内容が見えないので、list()で変換する
    # map(): イテレータの各要素に対して演算を加える
    print('map(): ', numlist, list(map(calc_double, numlist)))

    # filter()
    numlist = [1, 3, 6, 8]

    def even_three_div(x):
        return x % 3 == 0

    # filter(): イテレータの各要素に対して、条件に合致した要素だけに絞り込む
    print('filter(): ', numlist, list(filter(even_three_div, numlist)))

    # zip(): 同じ長さのリストに対して、要素番号が同じ物を合わせてタプルを作成する
    meallist = ['steak', 'salad', 'dessert', 'hogehge1', 'hogehoge333']
    drinklist = ['coffee', 'tea', 'water']
    print('zip(): ', list(zip(meallist, drinklist)))

    # dict(): キーワード引数を用いて辞書を作る
    print('dict(): ', dict(steak=1, salad='人参', dessert='プリン'))

    # dict()に与える、key-valueをzip()で作成する
    print('dict() + zip(): ', dict(zip(meallist, drinklist)))


# Python特有の応用機能
def execute_python_app_functions():
    # 無名関数を作る
    double_value = lambda x: x * 2
    result = double_value(2)
    pass


def main():
    execute_basic_python_functons()
    execute_python_app_functions()


if __name__ == '__main__':
    main()
