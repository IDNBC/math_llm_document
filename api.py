# 参考にした書籍[ディープラーニング実装入門 PyTorchによる画像・自然言語処理]に書かれていたコード
# Flaskを使って推論サーバーを構築する

# config: utf-8

from flask import Flask, jsonify

app = Flask(__name__)

@app.route('/')
def hello():
    ret = {'y': 3}
    return jsonify(ret)

# メイン関数
if __name__ == '__main__':
    app.run(debug=True)


# cdコマンドで作業ディレクトリへ移動後、python api.pyを実行
# IPアドレスなどが表示されたらサーバーが起動に成功

# 起動したサーバーのURLを設定する
import requests
url = 'http://<ipアドレス>:5000/'
# リクエストを送る
res = requests.get(url)
res
# 結果を表示
result = res.json()
result
result['y']


from flask import Flask, jsonify, requests
app = Flask(__name__)
# POSTメソッドに対応
@app.route('/', methods=['POST'])
def predict():
    # クエリの受け取り
    x = requests.json['query']
    ret = {'x': x}
    return jsonify(ret)

# メイン関数
if __name__ == '__main__':
    app.run(debug=True)


from sklearn.datasets import load_irisurl
url = 'http://<ipアドレス>:5000/'
# データの読み込み
x, t = load_iris(return_X_y=True)

query = list(x[0])
query, type(query)

# 辞書型にしておく
params = {'query': query}
params

# POSTリクエスト
res = requests.post(url, json=params)
res

# 結果の確認
result = res.json()
result


# 学習済みモデルの読み込み
from flask import Flask, jsonify, request
from datetime import datetime, timedelta
import azure.storage.blob as azureblob
import torch
import os

app = Flask(__name__)

# Blob Storageからネットワークと学習済みモデルダウンロード
model, pretrained = 'iris.py', 'iris.pt'
if not os.path.exists(model) or not os.path.exists(pretrained):
    print('downloading the model and the pretrained parameters')
    # Blobへ接続
    sas_token = azureblob.generate_account_sas(
        account_name = '<your-account-name>',
        account_key = '<your-account-key>',
        resource_types = azureblob.ResourceTypes(service=True),
        permission = azureblob.AccountSasPermissions(read=True),
        expiry = datetime.utcnow() + timedelta(hours=1)
    )
    blob_service_client = azureblob.BlobServiceClient(account_url='<your-account-url>', credential=sas_token)

    # コンテナとの接続
    conn_str = '<your-connection-string>'
    container = 'models'
    for filename in [model, pretrained]:
        with open(filename, 'wb') as my_blob:
            blob = azureblob.BlobClient.from_connection_string(conn_str=conn_str, container_name=container, blob_name=filename)
            download_stream = blob.download_blob()
            my_blob.write(download_stream.readall())

# ネットワークの定義と学習済みモデルのロード
from iris import Net
net = Net()
net.load_state_dict(torch.load(pretrained))

# POSTメソッドに対応
@app.route('/', methods=['POST'])
def predict():
    # クエリの受け取り
    x = requests.json['query']
    ret = {'x': x}
    return jsonify(ret)

# メイン関数
if __name__ == '__main__':
    app.run(debug=True)


from flask import Flask, jsonify, request
from datetime import datetime, timedelta
import azure.storage.blob as azureblob
import torch
import torch.nn.functional as F
import os

app = Flask(__name__)

# Blob Storageからネットワークのクラスと学習済みモデルをダウンロード
model, pretrained = 'iris.py', 'iris.pt'
if not os.path.exists(model) or not os.path.exists(pretrained):
    print('downloading the model and the pretrained parameters')
    # Blobへ接続
    sas_token = azureblob.generate_account_sas(
        account_name = '<your-account-name>',
        account_key = '<your-account-key>',
        resource_types = azureblob.ResourceTypes(service=True),
        permission = azureblob.AccountSasPermissions(read=True),
        expiry = datetime.utcnow() + timedelta(hours=1)
    )
    blob_service_client = azureblob.BlobServiceClient(account_url='<your-account-url>', credential=sas_token)

    # コンテナとの接続
    conn_str = '<your-connection-string>'
    container = 'models'
    for filename in [model, pretrained]:
        with open(filename, 'wb') as my_blob:
            blob = azureblob.BlobClient.from_connection_string(conn_str=conn_str, container_name=container, blob_name=filename)
            download_stream = blob.download_blob()
            my_blob.write(download_stream.readall())

# ネットワークの定義と学習済みモデルのロード
from iris import Net
net = Net()
net.load_state_dict(torch.load(pretrained))
net.eval()

# POSTメソッドに対応
@app.route('/', methods=['POST'])
def predict():
    # クエリの受け取り
    x = requests.json['query']
    x = torch.tensor(x).unsqueeze(0)
    # 推論
    with torch.no_grad():
        y = F.softmax(net(x), 1)[0]
        _, index = torch.max(y, 0)

    result = {'label': int(index), 'probability': float(y[index])}
    return jsonify(result)

# メイン関数
if __name__ == '__main__':
    app.run(debug=True)



import requestsfrom 
from sklearn.datasets import load_iris

# Irisデータセットの読み込み
x, t = load_iris(return_X_y=True)

# URLの指定
url = 'http://<IPアドレス>:5000/'

# 最初のサンプルをクエリに設定(リスト形式へ)
query = list(x[0])
query, type(query)

# 辞書型にしておく
params = {'query': query}
params

# POSTリクエスト
res = requests.post(url, json=params)
res

# 結果の確認
result = res.json()
result