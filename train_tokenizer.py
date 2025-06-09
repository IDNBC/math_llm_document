# トークナイザー訓練用コード math_tokenizer_v2用

import sentencepiece as spm

# # --- 設定 ---
# CORPUS_PATH = "corpus_for_tokenizer.txt"     # ステップ1で作成したファイル
# MODEL_PREFIX = "math_tokenizer_v2"           # 出力されるモデル名 (v2などバージョンが分かるように)
# VOCAB_SIZE = 200                             # 語彙サイズ (数字、演算子、一般的な数値パターンをカバーできる程度)
# MODEL_TYPE = 'bpe'                           # 'bpe' または 'unigram'

# # --- 特殊トークンの定義 ---
# # これらのIDは tokenizer_setup.py や model.py, train.py で使うIDと一致させる
# PAD_TOKEN = "<pad>"
# SOS_TOKEN = "<sos>"
# EOS_TOKEN = "<eos>"
# UNK_TOKEN = "<unk>"
# # ゼロ除算の答え "None" も一つの特殊トークンとして扱うと、より明確になる
# # (必須ではないが、推奨されるアプローチ)
# NONE_TOKEN = "<none>" 

# PAD_ID = 0
# SOS_ID = 1
# EOS_ID = 2
# UNK_ID = 3
# NONE_ID = 4 # 新しく追加

# print("トークナイザーの学習を開始します...")

# spm.SentencePieceTrainer.Train(
#     f'--input={CORPUS_PATH} '
#     f'--model_prefix={MODEL_PREFIX} '
#     f'--vocab_size={VOCAB_SIZE} '
#     f'--model_type={MODEL_TYPE} '
#     # character_coverage を 1.0 に設定して、全ての文字が語彙に含まれるようにする
#     f'--character_coverage=1.0 '
#     # 特殊トークンをユーザー定義シンボルとして登録 
#     # ここが原因でエラーが発生 ユーザ定義でSentencePieceが内部で管理する特別なトークンは含めていけない
#     f'--user_defined_symbols={PAD_TOKEN},{SOS_TOKEN},{EOS_TOKEN},{UNK_TOKEN},{NONE_TOKEN} '
#     # 特殊トークンのIDを明示的に指定
#     f'--pad_id={PAD_ID} '
#     f'--bos_id={SOS_ID} ' # <sos> は bos (begin of sentence)
#     f'--eos_id={EOS_ID} ' # <eos> は eos (end of sentence)
#     f'--unk_id={UNK_ID} '
#     # SentencePieceのデフォルトの動作を制御するオプション (推奨)
#     f'--control_symbols= ' # デフォルトで追加される制御シンボルを無効化
#     f'--byte_fallback=true ' # 未知のUTF-8文字をバイト列として扱うことで、<unk>を減らす
# )

# print(f"学習が完了しました。'{MODEL_PREFIX}.model' と '{MODEL_PREFIX}.vocab' が作成されました。")


import sentencepiece as spm

# --- 設定 ---
CORPUS_PATH = "corpus_for_tokenizer.txt"
MODEL_PREFIX = "math_tokenizer_v2"
VOCAB_SIZE = 300
MODEL_TYPE = 'bpe'

# --- 特殊トークンの定義 ---
# 文字列の定義はそのまま使用します
PAD_TOKEN = "<pad>"
SOS_TOKEN = "<sos>"
EOS_TOKEN = "<eos>"
UNK_TOKEN = "<unk>"
NONE_TOKEN = "<none>" # これはカスタムトークン

# IDの定義もそのまま使用します
PAD_ID = 0
SOS_ID = 1
EOS_ID = 2
UNK_ID = 3
# NONE_ID は user_defined_symbols 内で自動的に割り当てられます

print("トークナイザーの学習を開始します...")

# SentencePieceTrainer.Train に渡す引数を修正
spm.SentencePieceTrainer.Train(
    f'--input={CORPUS_PATH} '
    f'--model_prefix={MODEL_PREFIX} '
    f'--vocab_size={VOCAB_SIZE} '
    f'--model_type={MODEL_TYPE} '
    f'--character_coverage=1.0 '
    
    # --- ここからが修正箇所 ---
    # 1. 各特殊トークンのIDを指定
    f'--pad_id={PAD_ID} '
    f'--unk_id={UNK_ID} '
    f'--bos_id={SOS_ID} '
    f'--eos_id={EOS_ID} '
    
    # 2. 各特殊トークンの「文字列としての見た目」を指定
    f'--pad_piece={PAD_TOKEN} '
    f'--unk_piece={UNK_TOKEN} '
    f'--bos_piece={SOS_TOKEN} '
    f'--eos_piece={EOS_TOKEN} '
    
    # 3. --user_defined_symbols には、上記以外の「純粋なカスタムトークン」のみを指定
    # f'--user_defined_symbols={PAD_TOKEN},{SOS_TOKEN},{EOS_TOKEN},{UNK_TOKEN},{NONE_TOKEN} 'を修正した
    f'--user_defined_symbols={NONE_TOKEN} '
    
    # 4. その他の推奨オプション
    f'--byte_fallback=true '
)

print(f"学習が完了しました。'{MODEL_PREFIX}.model' と '{MODEL_PREFIX}.vocab' が作成されました。")