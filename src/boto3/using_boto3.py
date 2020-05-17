import boto3

# s3に接続して、バケット名を取得する
s3 = boto3.resource('s3')
for bucket in s3.buckets.all():
    print('s3 bucket name: ', bucket.name)

# Upload a new file
filename = 'IMG_0048.jpg'
data = open(filename, 'rb')
result = s3.Bucket('pybasic-s3').put_object(Key=filename, Body=data)

print('result: ', result)
