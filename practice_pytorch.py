import warnings
warnings.simplefilter("default")  # すべての警告を表示

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim # optimは最適化のためのパッケージ

print("torch version:", torch.__version__)

# 標準正規分布に従う入力
data = np.random.randn(200, 1)
# y=3xにノイズを追加した出力
label = 3*data + np.random.randn(200, 1)*0.5

x_train = data[:150, :]
y_train = label[:150, :]
x_test = data[:50, :]
y_test = label[:50, :]

# tesnsor化 : float32に変更
x_train = torch.tensor(x_train).float()
y_train = torch.tensor(y_train).float()
x_test = torch.tensor(x_test).float()
y_test = torch.tensor(y_test).float()

# fig, ax = plt.subplots()
# ax.scatter(x_train, y_train, alpha=0.8, label='train data')
# ax.scatter(x_test, y_test, alpha=0.8, label='test data')
# ax.set_xlabel(r'$x$', fontsize=20)
# ax.set_ylabel(r'$y$', fontsize=20)
# ax.legend() #凡例を付ける
# plt.show() #画面表示する

# 学習モデル定義(単回帰)
def model(x):
    return w*x + b # 入力xに重みwをかけ、バイアスbを足す

# 損失関数の定義 引数outputは予測値でyは正解値
def criterion(output, y):
    loss = ((output - y)**2).mean() #予測と正解の誤差を２乗したものの平均を取っている
    return loss

# パラメータの初期値
w = torch.tensor(0.0).float() # 32ビット不動小数点に
b = torch.tensor(0.0).float()

# 学習率 learning rateの略
lr = 0.01

# エポック数
num_epoch = 1000

train_loss_list = []
for epoch in range(num_epoch):
    # 予測
    output = model(x_train)
    # 損失関数を計算
    loss = criterion(output, y_train)
    # 勾配を計算
    grad_w = ((2*x_train)*(output-y_train)).mean()
    grad_b = (output-y_train).mean()
    # パラメータ更新
    w -= lr*grad_w
    b -= lr*grad_b
    grad = 0
    # lossを記録
    if (epoch%5==0):
        train_loss_list.append(loss)
        print(f'【EPOCH {epoch}】 loss : {loss:.5f}')


fig, ax = plt.subplots()
epoch_list = np.arange(0, 1000, 5)
ax.plot(epoch_list, train_loss_list)
ax.set_title(f'Result : w = {w:.4f}, b = {b:.4f}', fontsize=15)
ax.set_ylabel('train loss', fontsize=20)
plt.show()

# 予測値
pred = model(x_test)

fig, ax =plt.subplots()
ax.scatter(x_test, y_test, c='orange', alpha=0.5, label='test data')
ax.scatter(x_test, pred, c='blue', alpha=0.5, label='prediction')
ax.set_xlabel(r'$x$', fontsize=20)
ax.set_ylabel(r'$y$', fontsize=20)
ax.legend()
plt.show()

# 学習モデルを定義
def model(x):
    return w*x + b

# 損失関数を定義
def criterion(output, y):
    loss = ((output - y)**2).mean()
    return loss

# パラメータの初期値
# requires_grad=Trueとして自動微分を可能にする
w = torch.tensor(0.0, requires_grad=True).float()
b = torch.tensor(0.0, requires_grad=True).float()

# 学習率
lr = 0.01

# エポック数
num_epoch = 1000

train_loss_list = []
for epoch in range(num_epoch):
    # 予測
    output = model(x_train)
    # 損失関数を計算
    loss = criterion(output, y_train)
    # 勾配を計算 loss.backward()の1行の記述だけでよくなる
    loss.backward()

    with torch.no_grad():
        # パラメータ更新
        w -= lr*w.grad
        b -= lr*b.grad
        # 勾配の初期化
        w.grad.zero_()
        b.grad.zero_()

    # lossを記録
    if (epoch%5==0):
        train_loss_list.append(loss)
        print(f'【EPOCH {epoch}】 loss : {loss:.5f}')

fig, ax = plt.subplots(dpi=200)
epoch_list = np.arange(0, 1000, 5)
ax.plot(epoch_list, train_loss_list)
ax.set_title(f'Result : w = {w:.4f}, b = {b:.4f}', fontsize=15)
ax.set_ylabel('train loss', fontsize=20)
plt.show()

# 学習モデルを定義
def model(x):
    return w*x + b

# パラメータの初期値 自動微分on
w = torch.tensor(0.0, requires_grad=True).float()
b = torch.tensor(0.0, requires_grad=True).float()

# 損失関数を定義 平均二乗誤差でPyTorchには有名なものは搭載されていて呼び出すだけでいい
criterion = nn.MSELoss()

# 最適化手法を指定
optimizer = optim.SGD([w, b], lr=0.01)

# エポック数
num_epoch = 1000

train_loss_list = []
for epoch in range(num_epoch):
    # 勾配初期化
    optimizer.zero_grad()
    # 予測
    output = model(x_train)
    # 損失関数を計算
    loss = criterion(output, y_train)
    # 勾配を計算
    loss.backward()
    # パラメータ更新
    optimizer.step()

    # lossを記録
    if (epoch%5==0):
        train_loss_list.append(loss.detach().item())
        print(f'【EPOCH {epoch}】 loss : {loss:.5f}')

fig, ax = plt.subplots(dpi=200)
epoch_list = np.arange(0, 1000, 5)
ax.plot(epoch_list, train_loss_list)
ax.set_title(f'Result : w = {w:.4f}, b = {b:.4f}', fontsize=15)
ax.set_ylabel('train loss', fontsize=20)
plt.show()


"""
PyTorchは、深層学習を構成する線形回帰のようなネットワークや様々な変換を一つのブロックと見なして、
それを積み重ねることで簡単にネットワークが作成できます。その場合は、nnモジュールを使用します。
ここではNetクラスを宣言し、nn.Moduleを継承している。
"""
# 学習モデルを定義
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(1, 1, bias=True)

        # 初期値の設定   
        nn.init.constant_(self.fc.weight, 0.0)
        nn.init.constant_(self.fc.bias, 0.0)
        
    def forward(self, x):
        x = self.fc(x)
        return x

# インスタンス生成
model = Net()

# 損失関数を定義
criterion = nn.MSELoss()

# 最適化手法を決定
optimizer = optim.SGD(model.parameters(), lr=0.01)

# エポック数
num_epoch = 1000

train_loss_list = []
for epoch in range(num_epoch):
    # 訓練モードに変更
    model.train()
    # 勾配初期化
    optimizer.zero_grad()
    # 予測
    output = model(x_train)
    # 損失関数を計算
    loss = criterion(output, y_train)
    # 勾配を計算
    loss.backward()
    # パラメータ更新
    optimizer.step()

    # lossを記録
    if (epoch%5==0):
        train_loss_list.append(loss.detach().item())
        print(f'【EPOCH {epoch}】 loss : {loss.detach().item():.5f}')

fig, ax = plt.subplots()
epoch_list = np.arange(0, 1000, 5)
ax.plot(epoch_list, train_loss_list)
# w, bの確認の仕方が異なることに注意
ax.set_title(f'Result : w = {model.fc.weight.item():.4f}, b = {model.fc.bias.item():.4f}',
             fontsize=15)
ax.set_ylabel('train loss', fontsize=20)
plt.show()