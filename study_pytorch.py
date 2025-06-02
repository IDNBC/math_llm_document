import torch
import torch.nn as nn
from torch.nn import Transformer

# MathLLMクラスはnn.Moduleを継承している
""" nn.ModuleはPyTorchのニューラルネットワークの基本クラス
    レイヤーの定義、重みの保持、forwardメソッドの定義など土台を作っている
    重み(パラメータ)の登録と管理、モデルの保存と読み込み、ユーティリティの提供など
"""
class MathLLM(nn.Module):
    """ ここで初期化を行うコンストラクタ
        vocab_size=語彙数(トークン数)、d_model=各トークンの埋め込みベクトル次元数、
        nhead=マルチヘッドアテンションのヘッド数、num_layers=エンコーダ・デコーダのレイヤー数

        vocab_sizeはトークナイザーでトークン列に分割してID化するとき、トークンIDは0～(vocab_size - 1)
        に収まる。これをEmbedding層やLinear層で使う。
        vocab_size はトークナイザーに依存して決まり、事前に語彙表（Vocabulary）を構築する必要がある。
    """
    def __init__(self, vocab_size, d_model=128, nhead=4, num_layers=2):
        super().__init__() #親クラスのnn.Moduleのコンストラクタ呼び出し
        # Embedding層でトークン数を埋め込みベクトル次元数にする
        """ embedding層では整数のトークンIDを実数のベクトルに変換する層。
            トークンID(42)を受け取ったら、ランダムに初期化しd_model次元のベクトル(例：128次元)を返す。
            初期化はランダムでも、学習によって埋め込みベクトルは更新される。
            入力は [seq_len, batch_size] で、出力は [seq_len, batch_size, d_model]

            token_ids = torch.tensor([1, 5, 10])
            embedded = embedding(token_ids)
            のように使う。これで上記のテンソルという数式のトークンをベクトル空間上で学習できる。"""
        self.embedding = nn.Embedding(vocab_size, d_model)
        # Transformerモジュールでエンコーダとデコーダを指定のレイヤー数で構築
        """ これはTransformerモジュールで、
            nn.Transformer は エンコーダ・デコーダ構造の完全なTransformer を提供する高レベルAPI
            エンコーダはsrcを処理して意味表現を得る。
            デコーダはtgtをもとにエンコーダの出力を参照しながら予測を行う。
            この構造は入力→出力の変換タスクに使われる。
            ここではコンストラクタで引数になったd_model、nhead、num_layersが使われる。
            nn.Transformerは位置情報(Positional Encoding)を自動で処理しないため、通常は手動で追加する必要がある。
            教師あり学習時はtgtを1トークンずらした形式で与える(いわゆる教師強制)。
            """
        self.transformer = Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers
        )
        # 出力層 d_modelの出力を語彙数vocab_sizeに射影し、各トークンのスコア(logits)を得る
        """ Transformerの出力は各時刻で d_model 次元のベクトルになっている。
            このベクトルvocab_sizeの次元に変換することで、全トークンに対するスコア(logits)を得る。
            各トークン位置で「語彙中のどの単語が来る確率が高いか」を表す値が出力される。
            出力ベクトル [0.1, 0.8, -0.2, ...] → softmaxで最大値の位置が予測トークンになる。
            self.fcはlogits(スコア)を返すだけで、softmaxは通常は損失関数内部で適用される。
        """
        self.fc = nn.Linear(d_model, vocab_size)

    # ここはクラスの中に定義されたforwardメソッド
    # src(入力系列)とtgt(出力系列)を引数に取る
    """ srcはsourceのことで、モデルへの入力文にあたる。
        tgtはtargetのことで、生成してほしい出力文の一部。"""
    def forward(self, src, tgt):
        # src,tgtは [sequence_length, batch_size] の形
        """ 元のsrcは整数のテンソル [sequence_length, batch_size]
            self.embeddingに通すことで、実数値のベクトル表現 [sequence_length, batch_size, d_model] に変換
            tgtについても同様の処理をしている。"""
        src = self.embedding(src)
        tgt = self.embedding(tgt)
        # コンストラクタで定義したtransformerにsrc,tgtの引数を与えてoutputに代入
        """ srcからエンコーダの出力を得て、tgtから自己注意を通して、
            最終的に出力を計算し、outputに代入する。"""
        output = self.transformer(src, tgt)
        # 出力をfcメソッドに入れて返す
        # fcはfully connected(全結合層)の略。
        return self.fc(output)