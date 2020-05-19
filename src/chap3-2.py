# 定番アルゴリズムの検証

# 探索アルゴリズム


def search_linear(data, target):
    """線形にデータを探索する"""
    for i in range(len(data)):
        if data[i] == target:
            return i

    return -1


def binary_search(source, target):
    """二分探索"""
    start, end = 0, len(source)

    while start <= end:
        index = (start + end) // 2  # index: 中央値

        if source[index] == target:
            # 見つかったらそのindexを返す
            return index
        elif source[index] < target:
            # 検索範囲を後半部分にする
            start = index + 1
        else:
            # 検索範囲を前半部分にする
            end = index - 1

    # 見つからない
    return -1


def bubble_sort(source):
    """バブルソート"""

    def _swap_sort_if_needed(source, index):
        """指定されたインデックス以前の要素を、要素交換によって昇順にソートする

        :param source: データソース
        :param index: 何番目までソートするか
        """
        # 最も最後尾の比較対象から開始して、前に向かって比較していく
        # from: i-1から, to: -1まで
        for j in range(index - 1, -1, -1):
            front = source[j]
            back = source[j + 1]
            if front > back:
                # 隣り合う要素で、後の方が大きい場合、要素を入れ替える
                source[j], source[j + 1] = back, front
            else:
                # 隣り合う要素は、ソートされている
                break

    for i in range(0, len(source)):
        # 先頭要素から順番に、最後の要素まで、要素入れ替えでソートしていく
        _swap_sort_if_needed(source, index=i)


def main():
    # 線形探索、二分探索に渡すデータ。ソート済み。
    data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    target = 5

    # 線形探索
    result = search_linear(data, target)
    print('result1:', result)

    # 2分探索
    result = binary_search(source=data, target=target)
    print('result2:', result)

    # バブルソート
    bubble_sort_source = [1, 3, 250, 8, 7, 5, 10, 12, 1]
    bubble_sort(source=bubble_sort_source)
    print('bubble sort result:', bubble_sort_source)
    pass


if __name__ == '__main__':
    main()
