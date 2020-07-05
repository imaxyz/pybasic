"""
カレントの 'data' というディレクトリ配下における日本語のフォルダ名/ファイル名を、全て、ローマ字に変換する。
"""
import pykakasi
import os
import pprint
from typing import (
    Dict,
    List
)
kks = pykakasi.kakasi()


def convert_hiragana_to_romaji(kks, source_text: str) -> str:
    """
    Pykakasi + 独自のルールを用いて、日本語をローマ字に変換する。

    :param kks: PyKakasiオブジェクト
    :param source_text: 日本語の文字列
    :return: ローマ字に変換された文字列
    """
    text = source_text.replace('゙', '_dakuten_').replace('゚', '_handakten_')

    kks.setMode('H', 'a')   # Hiragana to ascii, default: no conversion
    kks.setMode('K', 'a')   # Katakana to ascii, default: no conversion
    kks.setMode('J', 'a')   # Japanese to ascii, default: no conversion
    # kks.setMode("r", "Hepburn")  # default: use Hepburn Roman table
    kks.setMode("s", True)  # add space, default: no separator
    # kks.setMode("C", True)  # capitalize, default: no capitalize

    converter = kks.getConverter()

    converted = converter.do(text)
    return converted.replace(' ', '_').replace("'", '-')


def create_file_paths(target_path: str) -> List[Dict[str, str]]:
    """
    ファイル名変換用の情報を格納したリストを生成する
    :param target_path: 走査するディレクトリパス
    :return: ファイル名変換用情報のリスト

        {
            old: 現在のファイルパス
            new: 新しいファイルパス
        }
    """
    result = []
    for root, dirs, files in os.walk(target_path):

        for file in files:
            conv_file = convert_hiragana_to_romaji(kks=kks, source_text=file)
            file_path = os.path.join(root, file)
            new_file_path = os.path.join(root, conv_file)

            result.append({
                'old': file_path,
                'new': new_file_path
            })

    return result


def create_dir_paths(target_path: str) -> List[Dict[str, str]]:
    """
    ディレクトリ名変換用の情報を格納したリストを生成する
    create_file_paths()と同様
    """
    result = []

    # 最下層のディレクトリから走査する
    for root, dirs, files in os.walk(target_path, topdown=False):
        for dir in dirs:
            dir_path = os.path.join(root, dir)
            conv_dir = convert_hiragana_to_romaji(kks=kks, source_text=dir)
            new_dir_path = os.path.join(root, conv_dir)

            result.append({
                'old': dir_path,
                'new': new_dir_path
            })

    return result


def execute_rename(specs: List[Dict[str, str]]):
    for dic in specs:
        try:
            os.renames(dic['old'], dic['new'])
        except Exception as e:
            print(e)
            pass


def main():

    file_paths = create_file_paths('data')
    dir_paths = create_dir_paths('data')

    # 日本語のファイル名を、ローマ字のファイル名に変換
    execute_rename(file_paths)
    print('files converted.')
    pprint.pprint(file_paths)

    # 日本語のディレクトリ名を、ローマ字のディレクトリ名に変換
    execute_rename(dir_paths)
    print('dirs converted.')
    pprint.pprint(dir_paths)

    print('done!')


if __name__ == '__main__':
    main()
