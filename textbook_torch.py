# 教科書のコードを書いて動かすファイルです。

import torch
import torch.nn as nn

inputs = torch.tensor(
    [[0.43, 0.15, 0.89],
    [0.55, 0.87, 0.66],
    [0.57, 0.85, 0.64],
    [0.22, 0.58, 0.33],
    [0.77, 0.25, 0.10],
    [0.05, 0.80, 0.55]]
)

query = inputs[1]

attn_scores_2 = torch.empty(inputs.shape[0])
for i, x_i in enumerate(inputs):
    attn_scores_2[i] = torch.dot(x_i, query)

print(attn_scores_2)

attn_weights_2_tmp = attn_scores_2 / attn_scores_2.sum()
print("Attention weights:", attn_weights_2_tmp)
print("Sum:", attn_weights_2_tmp.sum())

# ソフトマックス関数の基本実装
def softmax_naive(x):
    return torch.exp(x) / torch.exp(x).sum(dim=0)

attn_weights_2_naive = softmax_naive(attn_scores_2)
print("Attention weights:", attn_weights_2_naive)
print("Sum:", attn_weights_2_naive.sum())


attn_weights_2 = torch.softmax(attn_scores_2, dim=0)
print("Attention weights:", attn_weights_2)
print("Sum(use torch.softmax):", attn_weights_2.sum())

query = inputs[1]
context_vec_2 = torch.zeros(query.shape)
for i, x_i in enumerate(inputs):
    context_vec_2 += attn_weights_2[i]*x_i
print("context_vec_2:",context_vec_2)

attn_scores = torch.empty(6, 6)
for i, x_i in enumerate(inputs):
    for j, x_j in enumerate(inputs):
        attn_scores[i, j] = torch.dot(x_i, x_j) # 内積を取る
print("use torch:", attn_scores)

# ループ処理では遅いので、行列積を用いる
attn_scores = inputs @ inputs.T
print("use product:", attn_scores)

attn_weights = torch.softmax(attn_scores, dim=-1)
print("use torch.softmax:", attn_weights)

""" 
torch.softmax()関数のdimパラメータで入力テンソルのどの次元に沿って関数を計算するか指定する。
dim=-1を指定すると、引数に入れたテンソルの最後の次元に沿って正規化を適用するようにsoftmax関数に指示する。
"""

row_2_sum = sum([0.1385, 0.2379, 0.2333, 0.1240, 0.1082, 0.1581])
print("Row 2 sum:", row_2_sum)
print("All row sums:", attn_weights.sum(dim=-1))

all_context_vecs = attn_weights @ inputs
print(all_context_vecs)

"""
訓練可能な重み行列W_q、W_k、W_vを導入する。
この3つの行列は入力トークン埋め込みxをクエリベクトル、キーベクトル、値ベクトルに射影するのに使う。
"""

x_2 = inputs[1] # 2つ目の入力要素
d_in = inputs.shape[1] # 入力埋め込みのサイズ(d_in=3)
d_out = 2 # 出力埋め込みのサイズ(d_out=2)
# GPT型のモデルは入力と出力は同じ次元
# requires_gradはFalseなので、自動微分は無効化している
torch.manual_seed(123)
W_query = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_key = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_value = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)

query_2 = x_2 @ W_query
key_2 = x_2 @ W_key
value_2 = x_2 @ W_value
print("output query_2:",query_2)

keys = inputs @ W_key
values = inputs @ W_value
print("keys.shape:", keys.shape)
print("values.shape:", values.shape)

# Attentionスコアの計算
keys_2 = keys[1]
attn_score_22 = query_2.dot(keys_2)
print("Attention Score:", attn_score_22)

attn_scores_2 = query_2 @ keys.T
print(attn_scores_2)

d_k = keys.shape[-1]
attn_weights_2 = torch.softmax(attn_scores_2 / d_k**0.5, dim=-1) # 指数部0.5で平方根の意味
print("正規化後の重み:", attn_weights_2)

# コンテキストベクトルの計算 １つだけ
context_vec_2 = attn_weights_2 @ values
print(context_vec_2)

class SelfAttention_v1(nn.module):

    def __init__(self, d_in, d_out):
        super().__init__()
        self.W_query = nn.Parameter(torch.rand(d_in, d_out))
        self.W_key = nn.Parameter(torch.rand(d_in, d_out))
        self.W_value = nn.Parameter(torch.rand(d_in, d_out))
        
    def forward(self, x):
        keys = x @ self.W_key
        queries = x @ self.W_query
        values = x @ self.W_value

        attn_scores = queries @ keys.T
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1]**0.5, dim=-1
        )
        context_vec = attn_weights @ values
        return context_vec

