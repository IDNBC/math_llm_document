# math_tokenizer_corpus_expr_answer.txt

import sentencepiece as spm
import os

# --- 設定 ---
# このスクリプトが格納されているディレクトリをプロジェクトルートとする
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))

# 入力コーパスファイル名（fix_corpus.pyで生成された整形済みファイル）
# 前回のやり取りで「corpus_for_tokenizer.txt」に変更したので、ここもそれに合わせます。
INPUT_CORPUS_FILE_NAME = "math_tokenizer_corpus_expr_answer.txt"
INPUT_CORPUS_PATH = os.path.join(PROJECT_ROOT, INPUT_CORPUS_FILE_NAME)

# 出力するトークナイザーモデルのプレフィックス
# これにより、math_tokenizer_v3.model と math_tokenizer_v3.vocab が生成されます。
MODEL_PREFIX = os.path.join(PROJECT_ROOT, "math_tokenizer_v3") 

# トークナイザーの語彙サイズ
# 以前の300から、数学演算に特化して少し増やしてみます。
VOCAB_SIZE = 500 

# トークナイザーのモデルタイプ
# unigramはより頻度の高いまとまりをトークン化する傾向があります。
# 必要に応じて "bpe" も試せます。
MODEL_TYPE = "unigram" 

# --- SentencePiece トークナイザーの学習実行 ---
if __name__ == "__main__":
    print(f"Starting SentencePiece tokenizer training...")
    print(f"Input corpus: {INPUT_CORPUS_PATH}")
    print(f"Output model prefix: {MODEL_PREFIX}")
    print(f"Vocab size: {VOCAB_SIZE}")
    print(f"Model type: {MODEL_TYPE}")

    try:
        spm.SentencePieceTrainer.train(
            f'--input={INPUT_CORPUS_PATH} '
            f'--model_prefix={MODEL_PREFIX} '
            f'--vocab_size={VOCAB_SIZE} '
            f'--model_type={MODEL_TYPE} '
            f'--pad_id=0 --unk_id=3 --bos_id=1 --eos_id=2 ' # 特殊トークンIDは既存と合わせる
            '--hard_vocab_limit=false ' # 語彙サイズを厳密に守らず、必要に応じて増やす
            '--character_coverage=1.0 ' # 全ての文字が語彙に含まれるようにする
            '--train_extremely_large_corpus=true' # 大規模なコーパスでの学習を最適化
        )
        print(f"\nSentencePiece tokenizer training completed successfully!")
        print(f"New tokenizer model saved as {MODEL_PREFIX}.model and {MODEL_PREFIX}.vocab")
    except Exception as e:
        print(f"\nError during tokenizer training: {e}")
        print("Please ensure the input corpus file exists and is accessible.")