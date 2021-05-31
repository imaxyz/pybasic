def wrap_comments(func):
    def wrapper(*args, **kwargs):
        print(f'---- start   {func.__name__} -----')
        func(*args, **kwargs)
        print(f'\n---- end   {func.__name__} -----')

    return wrapper


@wrap_comments
def execute_basic_python_functons():

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


# global変数を定義
greeting = 'Good Morning'

# Python特有の応用機能


@wrap_comments
def execute_python_app_functions():
    # 無名関数を作る
    def double_value(x):
        x = x * 2
        return x

    print('double_value: ', double_value(2))

    # 無名関数を用いてソートする
    month_name = [(1, 'January'), (2, 'February'), (3, 'March')]
    month_name.sort(key=lambda x: x[1])
    print('month_name: ', month_name)

    # 任意引数を持つ関数を定義
    def variable_args(first, *args):
        print('variable_args: ', args)

    variable_args(1, 2, 3, 4, 5, 6, 66, 7)

    # ここで定義する変数は、global変数では無い。関数内スコープ変数。
    # greeting = 'Good Morning'

    def afternoon():
        # グローバル変数 greetingを関数内で使用
        global greeting
        greeting = 'Hello'
        print(greeting)

    afternoon()
    print(greeting)

    # Pythonで関数を実行するときは、「参照渡し」になる。（＝元の値が変更される）
    # しかしながら、数値/文字列に関しては「値渡し」になる（＝元の値は変更されない）
    # (!) Pythonでは、数値/文字列は、Immutableな存在。
    #   値を変更するときは、同じ変数が、新しい値の保管場所を参照するようになる。
    x = 1
    y = double_value(x)
    print('x, y:', x, y)

    # (!) Pythonでは、リスト/辞書は、mutableな存在。
    #   値を変更するときは、同じ変数が、同じ値の保管場所を参照する。
    def modify_list(original: list):
        original[1] = 'Apple'

    vegetables = ['Carrot', 'Potato', 'Pampkin']
    modify_list(vegetables)
    print('vegetables:', vegetables)

    pass


@wrap_comments
def execute_copy_function_demo():
    import copy

    # mutableなデータ型において単純な代入文を使うと同じ場所を指す
    list_a = [1, 2, 3]
    list_b = list_a
    list_b[1] = 100  # list_bに行った変更は、list_aにも及ぶ
    print('[1] list_a:', list_a, ', list_b:', list_b)

    # copy(shallow copy)でリストをコピーする
    list_c = [1, 2, 3]
    list_b = copy.copy(list_c)
    list_b[1] = 100
    print('[2] list_c:', list_c, ', list_b:', list_b)

    # 多次元リストをshallow copyするとコピーは完全では無い
    list_d = [[1, 2, 3], [4, 5, 6]]
    list_b = copy.copy(list_d)
    list_b[1][1] = 200
    print('[3] list_d:', list_d, ', list_b:', list_b)

    # 多次元リストのコピーには、deep copyを用いる
    list_e = [[1, 2, 3], [4, 5, 6]]
    list_b = copy.deepcopy(list_e)
    list_b[1][1] = 200
    print('[3] list_e:', list_e, ', list_b:', list_b)


@wrap_comments
def execute_generator():

    def count_up():
        """カウント値のジェネレータ関数
        yield文を使うことで、この関数がジェネレータ関数になる
        :return: カウント値
        """
        n = 1
        while True:
            yield n
            n += 1

    generator = count_up()

    for num in generator:
        print(num, end=' ')
        if num == 7:
            break

    for num in generator:
        print(num, end=' ')
        if num == 15:
            break
    pass


def main():
    execute_basic_python_functons()
    execute_python_app_functions()
    execute_copy_function_demo()
    execute_generator()
    pass


if __name__ == '__main__':
    main()
