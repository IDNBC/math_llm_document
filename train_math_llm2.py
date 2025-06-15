# import torch
# from torch.utils.data import Dataset, DataLoader
# from transformers import AutoConfig, AutoModelForCausalLM, get_linear_schedule_with_warmup
# import sentencepiece as spm
# import json
# from tqdm import tqdm
# import os
# import shutil
# import sys # sysモジュールをインポート

# # --- 設定 ---
# PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))

# # データセットのパス設定
# DATASET_FILE_NAME = "data_set/math_model_data_expr_answer.jsonl" # ★新しい整形済みデータセット名★
# DATASET_PATH = os.path.join(PROJECT_ROOT, DATASET_FILE_NAME)

# # トークナイザーのパス設定
# TOKENIZER_MODEL_NAME = "math_tokenizer_v3.model" # ★新しいトークナイザー名★
# TOKENIZER_MODEL_PATH = os.path.join(PROJECT_ROOT, TOKENIZER_MODEL_NAME)

# # モデルの保存パス
# OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output_model")
# MODEL_SAVE_PATH = os.path.join(OUTPUT_DIR, "lightweight_math_llm")

# # 学習パラメータ
# MODEL_NAME = None # 新規学習のためNone
# N_EMBD = 512       # 埋め込み次元数 (例: 256 -> 512)
# N_LAYER = 6        # レイヤー数 (例: 4 -> 6)
# N_HEAD = 8         # アテンションヘッド数 (例: 4 -> 8)
# VOCAB_SIZE = 500   # トークナイザーの語彙サイズと合わせる

# NUM_EPOCHS = 15      # エポック数 (例: 5 -> 15)
# BATCH_SIZE = 8       # バッチサイズ
# GRADIENT_ACCUMULATION_STEPS = 4 # 勾配蓄積ステップ数
# LEARNING_RATE = 3e-5 # 学習率 (例: 5e-5 -> 3e-5)
# WEIGHT_DECAY = 0.01
# WARMUP_STEPS = 0     # ウォームアップステップ数 (必要に応じて調整)

# # ★チェックポイント設定を追加★
# CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
# CHECKPOINT_SAVE_FREQ = 1 # エポックごとに保存する場合 (またはステップ数)
# CHECKPOINT_FILE_NAME = "checkpoint_epoch_{}.pth"
# LATEST_CHECKPOINT_FILE = os.path.join(CHECKPOINT_DIR, "latest_checkpoint.txt") # 最新チェックポイントのパスを記録するファイル

# # --- カスタムトークナイザーのラッパー (変更なし) ---
# class SentencePieceTokenizerWrapper:
#     def __init__(self, model_path):
#         self.sp_model = spm.SentencePieceProcessor()
#         self.sp_model.load(model_path)
#         self.unk_token_id = self.sp_model.unk_id()
#         self.bos_token_id = self.sp_model.bos_id()
#         self.eos_token_id = self.sp_model.eos_id()
#         self.pad_token_id = self.sp_model.pad_id()
#         self.vocab_size = self.sp_model.get_piece_size()
#         self._bos_token = self.sp_model.id_to_piece(self.bos_token_id)
#         self._eos_token = self.sp_model.id_to_piece(self.eos_token_id)
#         self._pad_token = self.sp_model.id_to_piece(self.pad_token_id)
#         self._unk_token = self.sp_model.id_to_piece(self.unk_token_id)

#     def encode(self, text, add_special_tokens=True):
#         ids = self.sp_model.encode_as_ids(text)
#         if add_special_tokens:
#             ids = [self.bos_token_id] + ids + [self.eos_token_id]
#         return ids

#     def decode(self, token_ids, skip_special_tokens=True):
#         if skip_special_tokens:
#             filtered_ids = [
#                 _id for _id in token_ids
#                 if _id not in [self.bos_token_id, self.eos_token_id, self.pad_token_id, self.unk_token_id]
#             ]
#             return self.sp_model.decode_ids(filtered_ids)
#         else:
#             return self.sp_model.decode_ids(token_ids)

#     @property
#     def pad_token(self):
#         return self._pad_token

#     @property
#     def bos_token(self):
#         return self._bos_token

#     @property
#     def eos_token(self):
#         return self._eos_token

#     @property
#     def unk_token(self):
#         return self._unk_token

#     def __len__(self):
#         return self.vocab_size


# # --- データセットの準備 ---
# class TextDataset(Dataset):
#     def __init__(self, file_path, tokenizer, max_length):
#         self.tokenizer = tokenizer
#         self.max_length = max_length
#         self.data = []

#         if not os.path.exists(file_path):
#             raise FileNotFoundError(f"Dataset file not found: {file_path}")

#         print(f"Loading dataset from {file_path}...")
#         with open(file_path, 'r', encoding='utf-8') as f:
#             for line in tqdm(f, desc="Loading data"):
#                 try:
#                     item = json.loads(line.strip())
#                     expr = item.get("expr")
#                     answer = item.get("answer")

#                     if expr is not None and answer is not None:
#                         # モデルに学習させたい入力形式
#                         text_to_train = f"{expr} = {answer}"
#                         self.data.append(text_to_train)
#                     else:
#                         print(f"Skipping line due to missing 'expr' or 'answer': {line.strip()}")
#                 except json.JSONDecodeError as e:
#                     print(f"Skipping malformed JSON line: {line.strip()} - Error: {e}")
#                 except Exception as e:
#                     print(f"An unexpected error occurred while processing line: {line.strip()} - Error: {e}")

#         print(f"Loaded {len(self.data)} data entries.")


#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         text = self.data[idx]
#         # トークナイザーでエンコード
#         # return_tensors='pt' は使用しない (カスタムトークナイザーのため)
#         # attention_mask は手動で作成
#         encoded = self.tokenizer.encode(text, add_special_tokens=True)
        
#         # パディングと切り捨て
#         input_ids = encoded[:self.max_length]
#         attention_mask = [1] * len(input_ids)
        
#         # パディング
#         padding_length = self.max_length - len(input_ids)
#         input_ids = input_ids + [self.tokenizer.pad_token_id] * padding_length
#         attention_mask = attention_mask + [0] * padding_length
        
#         return {
#             'input_ids': torch.tensor(input_ids, dtype=torch.long),
#             'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
#             'labels': torch.tensor(input_ids, dtype=torch.long) # CausalLMのため、labelsはinput_idsと同じ
#         }

# # --- メインの実行ブロック ---
# if __name__ == '__main__':
#     if sys.platform.startswith('win'):
#         # WindowsでDataLoaderのワーカープロセスがフリーズする問題対策
#         # mainスクリプトがimportされる際に多重に実行されるのを防ぐ
#         pass 

#     # デバイス設定
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"Using device: {device}")
#     if device.type == 'cuda':
#         print(f"GPU Name: {torch.cuda.get_device_name(0)}")
#         print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")

#     # ディレクトリ作成
#     os.makedirs(OUTPUT_DIR, exist_ok=True)
#     os.makedirs(CHECKPOINT_DIR, exist_ok=True) # ★チェックポイントディレクトリを作成★

#     # トークナイザーのロード
#     tokenizer = SentencePieceTokenizerWrapper(TOKENIZER_MODEL_PATH)
#     if tokenizer.vocab_size != VOCAB_SIZE:
#         print(f"Warning: Tokenizer vocab size ({tokenizer.vocab_size}) does not match config VOCAB_SIZE ({VOCAB_SIZE}). Using tokenizer's vocab_size.")
#         VOCAB_SIZE = tokenizer.vocab_size # トークナイザーの語彙サイズに合わせる

#     # モデルコンフィグの定義
#     config = AutoConfig.from_pretrained(
#         MODEL_NAME if MODEL_NAME else None, # Noneの場合、デフォルトコンフィグを使用
#         vocab_size=VOCAB_SIZE,
#         n_embd=N_EMBD,
#         n_layer=N_LAYER,
#         n_head=N_HEAD,
#         bos_token_id=tokenizer.bos_token_id,
#         eos_token_id=tokenizer.eos_token_id,
#         pad_token_id=tokenizer.pad_token_id,
#         unk_token_id=tokenizer.unk_token_id,
#         return_dict=True # 結果を辞書形式で返す
#     )

#     # モデルの初期化
#     model = AutoModelForCausalLM.from_config(config)
#     model.to(device)

#     print(f"Model parameters: {model.num_parameters() / 1e6:.2f}M")

#     # データセットとデータローダー
#     dataset = TextDataset(DATASET_PATH, tokenizer, max_length=config.max_position_embeddings)
#     dataloader = DataLoader(
#         dataset,
#         batch_size=BATCH_SIZE,
#         shuffle=True,
#         num_workers=0, # Windows環境では0推奨、Linuxでは適宜設定
#         pin_memory=True # GPUを使う場合に高速化
#     )

#     # オプティマイザとスケジューラ
#     optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
#     total_training_steps = len(dataloader) // GRADIENT_ACCUMULATION_STEPS * NUM_EPOCHS
#     scheduler = get_linear_schedule_with_warmup(
#         optimizer,
#         num_warmup_steps=WARMUP_STEPS,
#         num_training_steps=total_training_steps
#     )

#     # ★チェックポイントからの再開ロジックを追加★
#     start_epoch = 0
#     global_step = 0

#     def load_checkpoint():
#         # nonlocal start_epoch, global_step
#         if os.path.exists(LATEST_CHECKPOINT_FILE):
#             with open(LATEST_CHECKPOINT_FILE, 'r') as f:
#                 last_checkpoint_path = f.read().strip()
#             if os.path.exists(last_checkpoint_path):
#                 print(f"Loading checkpoint from {last_checkpoint_path}")
#                 checkpoint = torch.load(last_checkpoint_path, map_location=device)
#                 model.load_state_dict(checkpoint['model_state_dict'])
#                 optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#                 scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
#                 start_epoch = checkpoint['epoch'] + 1 # 次のエポックから開始
#                 global_step = checkpoint['global_step']
#                 print(f"Resuming training from epoch {start_epoch}, global step {global_step}")
#             else:
#                 print(f"No valid checkpoint found at {last_checkpoint_path}. Starting from scratch.")
#         else:
#             print("No latest checkpoint file found. Starting from scratch.")

#     def save_checkpoint(epoch, step):
#         checkpoint_path = os.path.join(CHECKPOINT_DIR, CHECKPOINT_FILE_NAME.format(epoch))
#         torch.save({
#             'epoch': epoch,
#             'global_step': step,
#             'model_state_dict': model.state_dict(),
#             'optimizer_state_dict': optimizer.state_dict(),
#             'scheduler_state_dict': scheduler.state_dict(),
#         }, checkpoint_path)
        
#         # 最新のチェックポイントパスをファイルに記録
#         with open(LATEST_CHECKPOINT_FILE, 'w') as f:
#             f.write(checkpoint_path)
#         print(f"Checkpoint saved to {checkpoint_path}")

#     load_checkpoint() # スクリプト開始時にチェックポイントをロード

#     # 学習ループ
#     print(f"\nStarting training for {NUM_EPOCHS} epochs from epoch {start_epoch}...")
#     model.train()
#     for epoch in range(start_epoch, NUM_EPOCHS):
#         print(f"\n--- Epoch {epoch+1}/{NUM_EPOCHS} ---")
#         epoch_loss = 0
#         for step, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}")):
#             input_ids = batch['input_ids'].to(device)
#             attention_mask = batch['attention_mask'].to(device)
#             labels = batch['labels'].to(device)

#             outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
#             loss = outputs.loss
#             loss = loss / GRADIENT_ACCUMULATION_STEPS # 勾配蓄積
#             loss.backward()
#             epoch_loss += loss.item()

#             if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
#                 optimizer.step()
#                 scheduler.step()
#                 optimizer.zero_grad()
#                 global_step += 1
        
#         avg_epoch_loss = epoch_loss / (len(dataloader) / GRADIENT_ACCUMULATION_STEPS) # 実際の更新回数で割る
#         print(f"Epoch {epoch+1} finished. Average Loss: {avg_epoch_loss:.4f}")

#         # エポックごとにチェックポイントを保存
#         if (epoch + 1) % CHECKPOINT_SAVE_FREQ == 0:
#             save_checkpoint(epoch, global_step)

#     print("\nTraining complete! Saving final model...")

#     # 最終モデルの保存
#     model.save_pretrained(MODEL_SAVE_PATH)
#     tokenizer.sp_model.save(TOKENIZER_MODEL_PATH) # トークナイザーも一緒に保存 (念のため)

#     print(f"Final model saved to {MODEL_SAVE_PATH}")
#     print(f"Tokenizer saved to {TOKENIZER_MODEL_PATH}")

#     # --- 数学能力の評価 (推論パート) ---
#     print("\n--- Loading trained model for math ability evaluation ---")

#     # 必ず最新のモデルをロードするようにする
#     # 最終保存パスからロード (または特定のチェックポイントからロードすることも可能)
#     loaded_model = AutoModelForCausalLM.from_pretrained(MODEL_SAVE_PATH)
#     loaded_model.to(device)
#     loaded_model.eval() # 推論モードに設定

#     loaded_tokenizer = SentencePieceTokenizerWrapper(TOKENIZER_MODEL_PATH)

#     # 試験的な問題リスト
#     test_prompts = [
#         "1 + 1 = ",
#         "10 * 5 = ",
#         "25 - 12 = ",
#         "100 / 4 = ",
#         "50 + 25 - 10 = ", # 複数項
#         "2 * (3 + 4) = ",  # カッコを含む
#         "-5 * 6 = ",       # 負の数
#         "1.5 + 2.3 = ",    # 小数点
#         "10 / 0 = ",       # ゼロ割
#         "99 * 99 = ",      # 2桁同士
#         "123 + 456 = ",    # 3桁の例 (学習データになければ難しい)
#     ]

#     print("\n--- Math Ability Evaluation ---")
#     for prompt in test_prompts:
#         input_ids = loaded_tokenizer.encode(prompt, add_special_tokens=True)
#         # BOSトークンのみで始める (プロンプト自体がシーケンスの開始)
#         # input_ids はすでに BOS と EOS を含んでいるので、そのまま使う
#         # ただし、推論時にはEOSは含めない方が自然な生成を促す
        
#         # プロンプトのエンコード (add_special_tokens=False でエンコードし、BOSだけ手動追加)
#         prompt_ids = loaded_tokenizer.encode(prompt, add_special_tokens=False)
#         input_ids = [loaded_tokenizer.bos_token_id] + prompt_ids
        
#         input_tensor = torch.tensor([input_ids]).to(device)

#         # モデルでテキストを生成
#         # max_lengthは生成する最大トークン数 (元の入力+生成部分)
#         # num_beamsはビームサーチのビーム数 (高品質な生成のため)
#         # early_stopping=True でEOSトークンで停止
#         output_sequences = loaded_model.generate(
#             input_ids=input_tensor,
#             max_length=loaded_tokenizer.max_length, # 最大長を適切に設定
#             num_beams=5, # 推論の品質向上
#             no_repeat_ngram_size=2,
#             early_stopping=True,
#             do_sample=False, # 常に同じ結果を得るため
#             temperature=1.0,
#             top_k=50,
#             top_p=1.0,
#             pad_token_id=loaded_tokenizer.pad_token_id, # パディングトークンIDを指定
#             eos_token_id=loaded_tokenizer.eos_token_id  # EOSトークンIDを指定
#         )

#         generated_text = loaded_tokenizer.decode(output_sequences[0].tolist(), skip_special_tokens=True)
        
#         # 生成されたテキストから、元のプロンプト部分を除去して解答のみを抽出
#         # generated_textはプロンプトを含むので、プロンプトの長さ分をスキップ
#         decoded_prompt = loaded_tokenizer.decode(input_ids, skip_special_tokens=True)
#         if generated_text.startswith(decoded_prompt):
#             answer_only = generated_text[len(decoded_prompt):].strip()
#         else:
#             answer_only = generated_text # プロンプトが含まれていない場合は全て解答と見なす

#         print(f"Q: {prompt.strip()} A: {answer_only}")



import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoConfig, AutoModelForCausalLM, get_linear_schedule_with_warmup
import sentencepiece as spm
import json
from tqdm import tqdm
import os
import shutil
import sys

# --- 設定 ---
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))

# データセットのパス設定
# math_model_data_expr_answer.jsonl が data_set/ フォルダ内にあることを前提
DATASET_FILE_NAME = "data_set/math_model_data_expr_answer.jsonl" 
DATASET_PATH = os.path.join(PROJECT_ROOT, DATASET_FILE_NAME)

# トークナイザーのパス設定
TOKENIZER_MODEL_NAME = "math_tokenizer_v3.model"
TOKENIZER_MODEL_PATH = os.path.join(PROJECT_ROOT, TOKENIZER_MODEL_NAME)

# モデルの保存パス
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output_model")
MODEL_SAVE_PATH = os.path.join(OUTPUT_DIR, "lightweight_math_llm") # 変更: モデル保存はディレクトリとして

# 学習パラメータ
# ★最新の推奨値とモデルサイズを適用★
MODEL_NAME = None # 新規学習のためNone
N_EMBD = 512       # 埋め込み次元数 (以前の256から増強)
N_LAYER = 6        # レイヤー数 (以前の4から増強)
N_HEAD = 8         # アテンションヘッド数 (以前の4から増強)
VOCAB_SIZE = 500   # トークナイザーの語彙サイズと合わせる (sentence3.pyで設定したもの)

NUM_EPOCHS = 15      # エポック数 (以前の5から増強)
BATCH_SIZE = 8       # バッチサイズ
GRADIENT_ACCUMULATION_STEPS = 4 # 勾配蓄積ステップ数
LEARNING_RATE = 3e-5 # 学習率 (以前の5e-5から調整)
WEIGHT_DECAY = 0.01
WARMUP_STEPS = 0     # ウォームアップステップ数
MAX_LENGTH = 128     # シーケンスの最大長 (変更なし、必要に応じて調整)
LOG_INTERVAL = 50    # ログ出力頻度 (変更なし)

# --- デバイス設定 ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
            # 特殊トークンを除外する際に、0（PAD）も除外することで、不要な空白を減らす
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

# --- データセットの準備 ---
class TextDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dataset file not found: {file_path}")

        print(f"Loading data from {file_path}...")
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Loading data"): # tqdmをファイル読み込みにも適用
                try:
                    item = json.loads(line.strip()) # .strip() を追加して空行や余分な空白に対応
                    expr = item.get("expr")
                    answer = item.get("answer")
                    
                    if expr is not None and answer is not None:
                        text_to_train = f"{expr} = {answer}" # モデルに学習させたい入力形式
                        self.data.append(text_to_train)
                    else:
                        # exprやanswerがない行はスキップ
                        pass
                except json.JSONDecodeError as e:
                    print(f"Skipping malformed JSON line: {line.strip()} - Error: {e}")
                    continue
                except Exception as e: # その他の予期せぬエラーをキャッチ
                    print(f"An unexpected error occurred while processing line: {line.strip()} - Error: {e}")
                    continue
        print(f"Loaded {len(self.data)} samples.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]
        encoded = self.tokenizer.encode(text, add_special_tokens=True)
        
        # 切り捨て
        input_ids = encoded[:self.max_length]
        
        # パディング (注意: attention_mask も必要だが、今回のモデルではlabelsと同じinput_idsを使用するためシンプルに)
        padding_length = self.max_length - len(input_ids)
        input_ids = input_ids + [self.tokenizer.pad_token_id] * padding_length
        
        input_ids_tensor = torch.tensor(input_ids, dtype=torch.long)
        labels = input_ids_tensor.clone() # CausalLMのため、labelsはinput_idsと同じ

        # Hugging Faceのモデルに渡すためにattention_maskも追加することを推奨
        attention_mask = [1] * (self.max_length - padding_length) + [0] * padding_length
        attention_mask_tensor = torch.tensor(attention_mask, dtype=torch.long)

        return {"input_ids": input_ids_tensor, "labels": labels, "attention_mask": attention_mask_tensor}


# --- メインの実行ブロック ---
if __name__ == '__main__':
    # WindowsでDataLoaderのnum_workersを使用するための設定 (必要に応じて)
    if sys.platform.startswith('win'):
        # num_workers > 0 でエラーが出る場合、コメントアウトを解除
        # torch.multiprocessing.set_start_method('spawn', force=True)
        pass 

    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")

    # ディレクトリ作成
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # トークナイザーのロード
    tokenizer = SentencePieceTokenizerWrapper(TOKENIZER_MODEL_PATH)
    # configのvocab_sizeとtokenizerのvocab_sizeを一致させる
    VOCAB_SIZE = tokenizer.vocab_size 
    print(f"Tokenizer loaded. Vocab size: {tokenizer.vocab_size}")
    print(f"BOS ID: {tokenizer.bos_token_id}, EOS ID: {tokenizer.eos_token_id}, PAD ID: {tokenizer.pad_token_id}, UNK ID: {tokenizer.unk_token_id}")

    # データセットとデータローダー
    dataset = TextDataset(DATASET_PATH, tokenizer, MAX_LENGTH)
    # num_workersはWindowsで問題が起きやすいので0に設定 (必要に応じて調整)
    # CPU数に合わせて増やしたい場合は os.cpu_count() などを使うが、まずは0で安全に
    dataloader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=0, # Windows環境では0を強く推奨。エラーが続くならここを0に
        pin_memory=torch.cuda.is_available() # GPUが利用可能ならTrue
    )
    print(f"DataLoader created with {len(dataloader)} batches.")

    # ★モデルコンフィグの定義を修正: from_pretrainedではなくゼロから作成★
    config = AutoConfig.for_model("gpt2") # GPT-2のようなベースモデルタイプを指定
    config.vocab_size = VOCAB_SIZE
    config.n_positions = MAX_LENGTH # max_position_embeddings の代わり
    config.n_embd = N_EMBD
    config.n_layer = N_LAYER
    config.n_head = N_HEAD
    config.bos_token_id = tokenizer.bos_token_id
    config.eos_token_id = tokenizer.eos_token_id
    config.pad_token_id = tokenizer.pad_token_id
    config.unk_token_id = tokenizer.unk_token_id
    config.return_dict = True
    config.architectures = ["GPT2LMHeadModel"] # 使用するモデルのアーキテクチャを明示
    config.gradient_checkpointing = True # メモリ節約のため有効に (Falseでも可)

    # モデルの初期化
    model = AutoModelForCausalLM.from_config(config)
    model.to(device)

    model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {model_params / 1e6:.2f}M (from scratch)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    num_training_steps = len(dataloader) // GRADIENT_ACCUMULATION_STEPS * NUM_EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=WARMUP_STEPS, num_training_steps=num_training_steps
    )

    print("Starting training...")
    model.train()
    global_step = 0
    total_loss = 0.0

    for epoch in range(NUM_EPOCHS):
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
        for step, batch in enumerate(progress_bar):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch["attention_mask"].to(device) # attention_mask を使用

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            loss = loss / GRADIENT_ACCUMULATION_STEPS
            loss.backward()

            total_loss += loss.item()

            if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # 勾配クリッピング
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % LOG_INTERVAL == 0:
                    avg_loss = total_loss / LOG_INTERVAL
                    print(f"Step {global_step}, Avg Loss: {avg_loss:.4f}")
                    total_loss = 0.0

        print(f"Epoch {epoch+1} finished. Total steps in epoch: {len(dataloader)}. Final Global Step: {global_step}")

    print("\nTraining completed!")

    # モデルとトークナイザーの最終保存
    model.save_pretrained(MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")

    # トークナイザーモデルファイルを直接コピー (Hugging Face形式で保存しない)
    shutil.copyfile(TOKENIZER_MODEL_PATH, os.path.join(MODEL_SAVE_PATH, TOKENIZER_MODEL_NAME))
    print(f"Tokenizer model copied to {MODEL_SAVE_PATH}")
    
    # Hugging Face形式での保存に必要なファイルを生成
    # SentencePieceTokenizerWrapperの情報を元にtokenizer_config.jsonとspecial_tokens_map.jsonを生成
    tokenizer_config_path = os.path.join(MODEL_SAVE_PATH, "tokenizer_config.json")
    special_tokens_map_path = os.path.join(MODEL_SAVE_PATH, "special_tokens_map.json")
    
    tokenizer_config = {
        "tokenizer_class": "SentencePieceTokenizerWrapper", # カスタムクラス名を指定
        "unk_token": tokenizer.unk_token,
        "bos_token": tokenizer.bos_token,
        "eos_token": tokenizer.eos_token,
        "pad_token": tokenizer.pad_token,
        "model_max_length": MAX_LENGTH,
        "vocab_size": tokenizer.vocab_size
    }
    with open(tokenizer_config_path, "w", encoding="utf-8") as f:
        json.dump(tokenizer_config, f, indent=4, ensure_ascii=False)
    print(f"Tokenizer config saved to {tokenizer_config_path}")

    special_tokens_map = {
        "unk_token": tokenizer.unk_token,
        "bos_token": tokenizer.bos_token,
        "eos_token": tokenizer.eos_token,
        "pad_token": tokenizer.pad_token
    }
    with open(special_tokens_map_path, "w", encoding="utf-8") as f:
        json.dump(special_tokens_map, f, indent=4, ensure_ascii=False)
    print(f"Special tokens map saved to {special_tokens_map_path}")


    print("\n--- Loading trained model for math ability evaluation ---")

    # 最終保存されたモデルをロードして評価
    loaded_model = AutoModelForCausalLM.from_pretrained(MODEL_SAVE_PATH)
    loaded_model.to(device)
    loaded_model.eval() # 推論モード

    # 保存されたトークナイザーモデルを使って再ロード
    loaded_tokenizer = SentencePieceTokenizerWrapper(os.path.join(MODEL_SAVE_PATH, TOKENIZER_MODEL_NAME))

    # 試験的な問題リスト (以前の例に新しいものを追加)
    math_prompts = [
        "2 + 3 =",
        "10 - 7 =",
        "4 * 5 =",
        "9 / 3 =",
        "12 + 8 - 5 =",
        "6 * (2 + 3) =",
        "15 / 3 + 2 =",
        "7 + 8 =",
        "20 - 4 =",
        "3 * 7 =",
        "8 / 2 =",
        "(4 + 6) * 2 =",
        "100 + 200 =",
        "500 - 120 =",
        "15 * 15 =",
        "123 / 3 =",
        "30 / 7 =", # 割り切れない場合
        "-5 + 8 =", # 負の数
        "7 - 10 =", # 負の数
        "0 * 100 =", # ゼロを含む
        "100 / 0 =" # ゼロ除算 (ZeroDivisionError)
    ]

    print("\n--- Evaluating Math Ability ---")
    for i, prompt in enumerate(math_prompts):
        input_prompt = f"{prompt} " # 最後にスペースを追加して、生成を促す
        print(f"\nPrompt {i+1}: '{input_prompt}'")
        
        # 推論用のエンコード (BOSのみ追加し、EOSは含めない)
        # generated_text がプロンプトから始まることを期待するため
        prompt_ids = loaded_tokenizer.encode(prompt, add_special_tokens=False)
        input_ids = [loaded_tokenizer.bos_token_id] + prompt_ids
        
        input_ids_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)

        with torch.no_grad():
            output_sequences = loaded_model.generate(
                input_ids=input_ids_tensor,
                max_length=MAX_LENGTH,
                num_beams=5,
                do_sample=False, # 推論時は再現性のためFalseが推奨
                top_k=50,
                temperature=1.0, # do_sample=False の場合、temperatureは影響しないことが多いが、念のため1.0
                no_repeat_ngram_size=2,
                early_stopping=True, # EOSトークンで停止
                pad_token_id=loaded_tokenizer.pad_token_id,
                eos_token_id=loaded_tokenizer.eos_token_id
            )

        generated_text = loaded_tokenizer.decode(output_sequences[0].tolist(), skip_special_tokens=True)
        
        # 生成されたテキストから、元のプロンプト部分を除去して解答のみを抽出
        # generated_textはプロンプトを含むので、プロンプトの長さ分をスキップ
        decoded_prompt_for_extraction = loaded_tokenizer.decode(input_ids, skip_special_tokens=True)
        if generated_text.startswith(decoded_prompt_for_extraction):
            answer_only = generated_text[len(decoded_prompt_for_extraction):].strip()
        else:
            answer_only = generated_text # プロンプトが含まれていない場合は全て解答と見なす (稀なケース)

        print(f"Generated: {answer_only}")

    print("\n--- Math evaluation completed ---")
    print("結果を確認し、モデルが正しく計算できているか目視で判断してください。")
    print("必要に応じて、学習エポック数を増やしたり、データセットの多様性を高めたりすることを検討してください。")