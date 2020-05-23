# 定番アルゴリズムの検証
import copy
from typing import (
    List
)


# 探索アルゴリズム


def linear_search(data, target):
    """線形探索. 計算量: O(n) """
    for i in range(len(data)):
        if data[i] == target:
            return i

    return -1


def binary_search(source, target):
    """二分探索. 計算量: O(log n) """
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
    """バブルソート. 計算量O:pow(n, 2)"""

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

    for i in range(0, len(source)):
        # 先頭要素から順番に、最後の要素まで、要素入れ替えでソートしていく
        _swap_sort_if_needed(source, index=i)


def insert_sort(source, gap=1):
    """挿入ソート. 計算量O:pow(n, 2))
    リストの最後の要素から、前に向かって、スワップを繰り返してソートする

    :param source: 操作対象
    :param gap: 比較の幅。挿入ソートを行う際の値の飛ばし方の量。
    :return: なし
    """
    for i in range(0, len(source)):
        for j in range(i - gap, -1, -gap):
            if source[j] > source[j + 1]:

                source[j], source[j + 1] = source[j + 1], source[j]
            else:
                break   # バブルソートと比較してこの処理が異なる


def selection_sort(source):
    """選択ソート. 計算量: paw(n, 2) """

    # 最初から順番に整列させてゆく
    for i in range(0, len(source) - 1):
        min_i = i

        # 最小値の探索を行う
        for j in range(i + 1, len(source)):
            if source[min_i] > source[j]:
                min_i = j

        # 最小値と、捜査中の要素を交換する
        source[min_i], source[i] = source[i], source[min_i]


def shell_sort(source):
    """シェルソート. 計算量:O(n log n)"""
    gaps = [7, 3, 1]  # ギャップの値をあらかじめ設定
    for gap in gaps:  # gapを段々狭めて繰り返す
        insert_sort(source, gap)


def heap_sort(source):
    """ヒープソート. 計算量: O(n log n)
    - Pythonの標準ライブラリを使って、ヒープソートする
    - ここでいう「ヒープ」とは、根が全要素の中で最小値(or最大値)になっている2分木のこと。
    """
    from heapq import heappush, heappop  # ヒープを扱う標準ライブラリをインポート

    heap = []

    # ヒープに全データを追加
    while source:
        heappush(heap, source.pop())

    # ヒープから最小値を取り出して，データソースの最後に追加していく
    while heap:
        # heappop()で、ヒープから最小値を取り出す
        source.append(heappop(heap))


def quick_sort(source) -> List[int]:
    """クイックソート. 計算量: O(n log n)"""
    if len(source) <= 1:
        return source

    n = len(source)

    # いったん中央の値を求める
    median_value = source[n // 2]

    # 分割する領域を作成
    smalls, bigs, middles = [], [], []

    # データソースの値を、3つの領域に振り分ける
    for i in range(n):
        if source[i] < median_value:
            smalls.append(source[i])
        elif source[i] > median_value:
            bigs.append(source[i])
        else:
            middles.append(source[i])

    # 大小が発生した領域を更に分割する
    if smalls:
        smalls = quick_sort(smalls)

    if bigs:
        bigs = quick_sort(bigs)

    # 極限まで分割した各領域を合成する
    return smalls + middles + bigs


def merge_sort(source) -> List[int]:
    """マージソート. 計算量: O(n log n) """
    if len(source) <= 1:
        return source

    # 中央の要素番号を取得
    median_index = len(source) // 2

    # 前半分の要素を取得
    lefts = merge_sort(source[:median_index])
    left_index = 0

    # 後ろ半分の要素を取得
    rights = merge_sort(source[median_index:])
    right_index = 0

    # マージ領域を作成
    merged = []

    # 前半分の要素と、後ろ半分の要素を比較して、小さい順に、要素をマージ領域に追加する
    # 結果: mergedは、要素が昇順にソートされた集合になる
    while left_index < len(lefts) and right_index < len(rights):

        if lefts[left_index] <= rights[right_index]:
            merged.append(lefts[left_index])
            left_index += 1
        else:
            merged.append(rights[right_index])
            right_index += 1

    # マージ後、前半分領域と、後ろ半分領域で、データが余っていたら、マージ領域の末尾に追加
    if left_index < len(lefts):
        merged.extend(lefts[left_index:])
    elif right_index < len(rights):
        merged.extend(rights[right_index:])

    # ソート済みのマージされた要素群を返す
    return merged


def main():
    # 線形探索、二分探索に渡すデータ。ソート済み。
    data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    target = 5

    # 線形探索
    result = linear_search(data, target)
    print('linear search result:', result)

    # 2分探索
    result = binary_search(source=data, target=target)
    print('binary search result:', result)

    # バブルソート
    bubble_sort_source = [1, 3, 250, 8, 7, 5, 10, 12, 1]
    bubble_sort(source=bubble_sort_source)
    print('bubble sort result:', bubble_sort_source)

    # 挿入ソート
    insert_sort_source = [1, 3, 250, 8, 7, 5, 10, 12, 1]
    insert_sort(source=insert_sort_source)
    print('insert_sort result:', insert_sort_source)

    # 選択ソート
    selection_sort_source = [1, 3, 250, 8, 7, 5, 10, 12, 1]
    selection_sort(source=selection_sort_source)
    print('selection_sort result:', selection_sort_source)

    # シェルソート
    shell_sort_source = [1, 3, 250, 8, 7, 5, 10, 12, 1]
    shell_sort(source=shell_sort_source)
    print('shell_sort result:', shell_sort_source)

    # ヒープソート
    heap_sort_source = [1, 3, 250, 8, 7, 5, 10, 12, 1]
    heap_sort(source=heap_sort_source)
    print('heap_sort result:', heap_sort_source)

    # クイックソート（再帰処理なので元のデータソースを変更しない）
    quick_sort_source = [1, 3, 250, 8, 7, 5, 10, 12, 1]
    # quick_sort_source = [250]
    quick_sort_result = quick_sort(source=quick_sort_source)
    print('quick_sort result:', quick_sort_result)

    # マージソート（再帰処理なので元のデータソースを変更しない）
    merge_sort_source = [1, 3, 250, 8, 7, 5, 10, 12, 1]
    merge_sort_result = merge_sort(source=merge_sort_source)
    print('merge_sort result:', merge_sort_result)


if __name__ == '__main__':
    main()
