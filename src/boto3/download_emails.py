"""簡易メールダウンロードツール
# 仕様
- S3のメールバケットから全てのメールファイルのリストを取得する
  - メールファイルに拡張子「.eml」を付加して保存パスを作成する
  - 保存パスに同名のファイルが存在しない場合、メールをS3から保存パスにダウンロードする
"""
import os
import boto3
from pathlib import Path
from typing import (
    List,
    Tuple,
)


def execute_setting():
    """localの.evnを環境変数としてロードする"""

    from dotenv import load_dotenv
    load_dotenv()

    # OR, the same with increased verbosity
    load_dotenv(verbose=True)

    # OR, explicitly providing path to '.env'
    from pathlib import Path  # Python 3.6+ only
    env_path = Path('.') / '.env'
    load_dotenv(dotenv_path=env_path)


def save_s3_mails(bucket_name: str,
                  save_dir_name: str) -> Tuple[List[str], List[str]]:
    """
    s3からメールをダウンロードして保存する。
    事前条件: s3のなんらかのバケットにメール受信設定がなされていること

    :param bucket_name: バケット名
    :param save_dir_name:  メールを保存するlocalパス
    :return:
        saved: 保存したメールパスのリスト
        skipped: 保存しなかったメールパスのリスト
    """

    # 書き込みフォルダの存在確認
    folder_path = Path(save_dir_name)
    if not folder_path.exists():
        message = f"Error: '{folder_path.name}' folder is not found."
        raise Exception(message)

    # S3のリソースを取得
    s3_resource = boto3.resource('s3')

    # S3の通信クライアントを取得
    s3_client = boto3.client('s3')

    # S3リソースからバケット内のファイルを取得する
    bucket = s3_resource.Bucket(bucket_name)
    s3_objects = bucket.objects.filter(Prefix='')

    saved = []      # 保存したパスのリスト
    skiped = []     # 保存しなかったパスのリスト
    for s3_object in s3_objects:
        mail_name = s3_object.key

        # 保存パスを作成
        save_path = f'{folder_path.name}/{mail_name}.eml'

        # 書き込み予定のPathを調べて、存在しなければ書き込む
        if not Path(save_path).exists():

            # S3クライアントにダウンロードを要求
            s3_client.download_file(bucket_name, mail_name, save_path)
            saved.append(save_path)
        else:
            skiped.append(save_path)

    return saved, skiped


def main():
    execute_setting()

    # 環境変数からバケット名を取得する
    s3_mail_bucket_name = os.getenv('S3_MAIL_BUCKET_NAME')

    # S3からメールをダウンロードする
    saved, skiped = save_s3_mails(s3_mail_bucket_name, 'mails')

    # 結果を出力する
    print('--------[result]-----------')
    for saved_path in saved:
        print('* saved: ' + saved_path)

    print('--------[skipped]-----------')
    for skiped_path in skiped:
        print('- skipped: ' + skiped_path)


if __name__ == '__main__':
    main()
