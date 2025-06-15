import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoConfig, AutoModelForCausalLM, get_linear_schedule_with_warmup
import sentencepiece as spm
import json
from tqdm import tqdm
import os
import shutil
import sys # sysモジュールを追加

# --- 設定 (変更なし) ---
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
TOKENIZER_MODEL_NAME = "math_tokenizer_v3.model"
DATASET_FILE_NAME = "data_set/math_model_data_expr_answer.jsonl"
TOKENIZER_MODEL_PATH = os.path.join(PROJECT_ROOT, TOKENIZER_MODEL_NAME)
DATASET_PATH = os.path.join(PROJECT_ROOT, DATASET_FILE_NAME)

MODEL_NAME = "lightweight_math_llm"
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output_model")
MAX_LENGTH = 128
BATCH_SIZE = 8
GRADIENT_ACCUMULATION_STEPS = 4
LEARNING_RATE = 5e-5
NUM_EPOCHS = 5
WEIGHT_DECAY = 0.01
WARMUP_STEPS = 0
LOG_INTERVAL = 50

# --- デバイス設定 (変更なし) ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- カスタムトークナイザーのラッパー (変更なし) ---
class SentencePieceTokenizerWrapper:
    # ... (既存のコードをそのまま貼り付け) ...
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

# --- データセットの準備 (変更なし) ---
class TextDataset(Dataset):
    # ... (既存のコードをそのまま貼り付け) ...
    def __init__(self, file_path, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []
        print(f"Loading data from {file_path}...")
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f):
                try:
                    item = json.loads(line)
                    expr = item.get("expr")
                    answer = item.get("answer")
                    
                    if expr is not None and answer is not None:
                        text_to_train = f"{expr} = {answer}"
                        self.data.append(text_to_train)
                    else:
                        pass
                except json.JSONDecodeError as e:
                    print(f"Skipping malformed JSON line: {line.strip()} - Error: {e}")
                    continue
        print(f"Loaded {len(self.data)} samples.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]
        encoded = self.tokenizer.encode(text, add_special_tokens=True)
        input_ids = encoded[:self.max_length]
        padding_length = self.max_length - len(input_ids)
        input_ids = input_ids + [self.tokenizer.pad_token_id] * padding_length
        input_ids_tensor = torch.tensor(input_ids, dtype=torch.long)
        labels = input_ids_tensor.clone()
        return {"input_ids": input_ids_tensor, "labels": labels}

# --- メインの実行ブロック ---
if __name__ == '__main__':
    # WindowsでDataLoaderのnum_workersを使用するための設定
    # これをmainブロックの最初に置くことで、子プロセスが適切に起動される
    # multiprocessing.freeze_support() は exe 化する場合に必要だが、ここでは念のため
    # sys.platform.startswith('win') は Windows のみで実行するためのチェック
    if sys.platform.startswith('win'):
        # Python 3.8以降では'spawn'がデフォルトだが、明示的に設定するのも良い
        # torch.multiprocessing.set_start_method('spawn', force=True) # これを有効にするとエラーが解消されることが多い
        pass # 現状、multiprocessing.spawn.py が自動で呼び出しているためコメントアウト

    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")

    tokenizer = SentencePieceTokenizerWrapper(TOKENIZER_MODEL_PATH)
    print(f"Tokenizer loaded. Vocab size: {tokenizer.vocab_size}")
    print(f"BOS ID: {tokenizer.bos_token_id}, EOS ID: {tokenizer.eos_token_id}, PAD ID: {tokenizer.pad_token_id}, UNK ID: {tokenizer.unk_token_id}")

    dataset = TextDataset(DATASET_PATH, tokenizer, MAX_LENGTH)
    # num_workers は環境によって調整。0 にするとマルチプロセスを使わず、エラーを回避できるが遅くなる
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=os.cpu_count() // 2 or 1, pin_memory=True)
    print(f"DataLoader created with {len(dataloader)} batches.")

    config = AutoConfig.from_pretrained(
        "gpt2",
        vocab_size=tokenizer.vocab_size,
        n_positions=MAX_LENGTH,
        n_embd=256,
        n_layer=4,
        n_head=4,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        gradient_checkpointing=True
    )

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

            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss

            loss = loss / GRADIENT_ACCUMULATION_STEPS
            loss.backward()

            total_loss += loss.item()

            if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
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

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Saving model and tokenizer to {OUTPUT_DIR}...")

    model_save_path = os.path.join(OUTPUT_DIR, MODEL_NAME)
    model.save_pretrained(model_save_path)
    print(f"Model saved to {model_save_path}")

    tokenizer_save_path = os.path.join(OUTPUT_DIR, TOKENIZER_MODEL_NAME)
    shutil.copyfile(TOKENIZER_MODEL_PATH, tokenizer_save_path)
    print(f"Tokenizer model saved to {tokenizer_save_path}")

    tokenizer_config_path = os.path.join(OUTPUT_DIR, "tokenizer_config.json")
    special_tokens_map_path = os.path.join(OUTPUT_DIR, "special_tokens_map.json")

    tokenizer_config = {
        "tokenizer_class": "SentencePieceTokenizerWrapper",
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

    loaded_model = AutoModelForCausalLM.from_pretrained(model_save_path)
    loaded_model.to(device)
    loaded_model.eval()

    loaded_tokenizer = SentencePieceTokenizerWrapper(tokenizer_save_path)

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
        "(4 + 6) * 2 ="
    ]

    print("\n--- Evaluating Math Ability ---")
    for i, prompt in enumerate(math_prompts):
        input_prompt = f"{prompt} "
        print(f"\nPrompt {i+1}: '{input_prompt}'")
        input_ids = loaded_tokenizer.encode(input_prompt, add_special_tokens=True)
        input_ids_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)

        with torch.no_grad():
            output_sequences = loaded_model.generate(
                input_ids=input_ids_tensor,
                max_length=MAX_LENGTH,
                num_beams=5,
                do_sample=True,
                top_k=50,
                temperature=0.7,
                no_repeat_ngram_size=2,
                early_stopping=True,
                pad_token_id=loaded_tokenizer.pad_token_id,
                eos_token_id=loaded_tokenizer.eos_token_id
            )

        generated_text = loaded_tokenizer.decode(output_sequences[0].tolist(), skip_special_tokens=True)
        generated_answer = generated_text.replace(input_prompt, "").strip()
        print(f"Generated: {generated_answer}")

    print("\n--- Math evaluation completed ---")
    print("結果を確認し、モデルが正しく計算できているか目視で判断してください。")
    print("必要に応じて、学習エポック数を増やしたり、データセットの多様性を高めたりすることを検討してください。")