import sentencepiece as spm
import os
import sys
import json # TextDataset内でjsonを使う場合があるため、念のため残す
# tqdm はトークナイザのテスト自体には不要ですが、元のTextDatasetのコードで使われているので、
# もしTextDatasetをコピーするなら必要になる場合があります。ここではコメントアウト。
# from tqdm import tqdm 

# --- 設定 ---
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
TOKENIZER_MODEL_NAME = "math_tokenizer_v2.model"
TOKENIZER_MODEL_PATH = os.path.join(PROJECT_ROOT, TOKENIZER_MODEL_NAME)

# --- カスタムトークナイザーのラッパー ---
class SentencePieceTokenizerWrapper:
    def __init__(self, model_path):
        self.sp_model = spm.SentencePieceProcessor()
        self.sp_model.load(model_path)
        self.unk_token_id = self.sp_model.unk_id()
        self.bos_token_id = self.sp_model.bos_id()
        self.eos_token_id = self.sp_model.eos_id()
        self.pad_token_id = self.sp_model.pad_id()
        self.vocab_size = self.sp_model.get_piece_size()
        self._bos_token = self.sp_model.id_to_piece(self.bos_token_id)
        self._eos_token = self.sp_model.id_to_piece(self.eos_token_id)
        self._pad_token = self.sp_model.id_to_piece(self.pad_token_id)
        self._unk_token = self.sp_model.id_to_piece(self.unk_token_id)

    def encode(self, text, add_special_tokens=True):
        ids = self.sp_model.encode_as_ids(text)
        if add_special_tokens:
            ids = [self.bos_token_id] + ids + [self.eos_token_id]
        return ids

    def decode(self, token_ids, skip_special_tokens=True):
        if skip_special_tokens:
            filtered_ids = [
                _id for _id in token_ids
                if _id not in [self.bos_token_id, self.eos_token_id, self.pad_token_id, self.unk_token_id]
            ]
            return self.sp_model.decode_ids(filtered_ids)
        else:
            return self.sp_model.decode_ids(token_ids)

    @property
    def pad_token(self):
        return self._pad_token

    @property
    def bos_token(self):
        return self._bos_token

    @property
    def eos_token(self):
        return self._eos_token

    @property
    def unk_token(self):
        return self._unk_token

    def __len__(self):
        return self.vocab_size

# --- メインの実行ブロック ---
if __name__ == '__main__':
    # デバイス設定はトークナイザーテストでは不要なので削除またはコメントアウト
    # import torch # もしGPU名などを表示したいなら必要
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(f"Using device: {device}")
    # if device.type == 'cuda':
    #     print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    #     print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")

    tokenizer = SentencePieceTokenizerWrapper(TOKENIZER_MODEL_PATH)
    print(f"Tokenizer loaded. Vocab size: {tokenizer.vocab_size}")
    print(f"BOS ID: {tokenizer.bos_token_id}, EOS ID: {tokenizer.eos_token_id}, PAD ID: {tokenizer.pad_token_id}, UNK ID: {tokenizer.unk_token_id}")

    # --- トークナイザーの挙動を確認 ---
    test_texts = [
        "2 + 3 = 5",
        "10 - 7 = 3",
        "4 * 5 = 20",
        "9 / 3 = 3",
        "-38 * 9 = -342",
        "0.5 + 1.2 = 1.7",
        "\\frac{1}{2}", # LaTeX形式の例
        "$0$",         # LaTeX形式の例
        "問題: 2 + 3 答え: 5" # 新しい入出力形式を試す場合
    ]

    print("\n--- Tokenizer Test ---")
    for text in test_texts:
        ids = tokenizer.encode(text, add_special_tokens=False) # 特殊トークンなしでエンコード
        decoded_text = tokenizer.decode(ids, skip_special_tokens=False) # デコード時も特殊トークンをスキップしない
        pieces = [tokenizer.sp_model.id_to_piece(id_) for id_ in ids]
        print(f"Original: '{text}'")
        print(f"IDs:      {ids}")
        print(f"Pieces:   {pieces}")
        print(f"Decoded:  '{decoded_text}'")
        print("-" * 30)
    print("--- Tokenizer Test End ---\n")

    sys.exit(0)