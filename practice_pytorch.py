import warnings
warnings.simplefilter("default")  # すべての警告を表示

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim # optimは最適化のためのパッケージ

print("torch version:", torch.__version__)

# # 標準正規分布に従う入力
# data = np.random.randn(200, 1)
# # y=3xにノイズを追加した出力
# label = 3*data + np.random.randn(200, 1)*0.5

# x_train = data[:150, :]
# y_train = label[:150, :]
# x_test = data[:50, :]
# y_test = label[:50, :]

# # tesnsor化 : float32に変更
# x_train = torch.tensor(x_train).float()
# y_train = torch.tensor(y_train).float()
# x_test = torch.tensor(x_test).float()
# y_test = torch.tensor(y_test).float()

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