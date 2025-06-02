import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import math
import os # チェックポイント保存用

from tokenizer_setup import load_tokenizer, get_vocab_size, PAD_ID, TOKENIZER_MODEL_PATH
from dataset import ArithmeticDataset, collate_fn
from model import Seq2SeqTransformer

# --- 設定 ---
# データセットのパス
DATASET_PATH = "dataset_all_combined3.jsonl" # ★★★ ご自身のデータセットパスに変更 ★★★

# モデルパラメータ (model.pyのテスト時と同じ値を初期値として使用)
EMB_SIZE = 128
NHEAD = 4
FFN_HID_DIM = 256
NUM_ENCODER_LAYERS = 2
NUM_DECODER_LAYERS = 2
DROPOUT = 0.1
MAX_SEQ_LEN_DATA = 50 # データセットの最大長 (dataset.pyのmax_lengthと合わせる)
MAX_SEQ_LEN_MODEL = 50 # モデルのPositionalEncodingのmax_len (上記と同じでよい)


# 学習パラメータ
LEARNING_RATE = 0.0001 # Transformerでは小さめの学習率が推奨されることが多い
BATCH_SIZE = 32 # GPUメモリに応じて調整
NUM_EPOCHS = 20    # 学習エポック数
CLIP_GRAD = 1.0   # 勾配クリッピングの値 (しない場合はNone)

# その他
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_DIR = "checkpoints"
CHECKPOINT_SAVE_INTERVAL = 1 # エポックごと
LOG_INTERVAL = 50 # バッチごと

def train_epoch(model, dataloader, optimizer, criterion, vocab_size, epoch_num):
    model.train() # 学習モード
    total_loss = 0
    start_time = time.time()

    for i, batch in enumerate(dataloader):
        src = batch['src'].to(DEVICE)                     # (batch_size, src_len)
        tgt_input = batch['tgt_input'].to(DEVICE)         # (batch_size, tgt_len)
        tgt_label = batch['tgt_label'].to(DEVICE)         # (batch_size, tgt_len)
        src_padding_mask = batch['src_padding_mask'].to(DEVICE) # (batch_size, src_len)
        tgt_padding_mask = batch['tgt_padding_mask'].to(DEVICE) # (batch_size, tgt_len)
        
        optimizer.zero_grad()

        # モデルからの出力 (logits)
        # memory_key_padding_mask には src_padding_mask を使用
        logits = model(src, tgt_input, src_padding_mask, tgt_padding_mask, memory_key_padding_mask=src_padding_mask)
        # logits: (batch_size, tgt_len, vocab_size)
        
        # 損失計算
        # criterion (CrossEntropyLoss) は (N, C) の入力を期待する
        # N = batch_size * tgt_len, C = vocab_size
        # tgt_label も (N) に変形
        loss = criterion(logits.reshape(-1, vocab_size), tgt_label.reshape(-1))
        
        loss.backward()

        if CLIP_GRAD is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD)
        
        optimizer.step()
        
        total_loss += loss.item()

        if (i + 1) % LOG_INTERVAL == 0:
            elapsed_batch = time.time() - start_time
            print(f"| Epoch {epoch_num+1:3d} | Batch {i+1:5d}/{len(dataloader):5d} "
                  f"| lr {optimizer.param_groups[0]['lr']:.2e} "
                  f"| ms/batch {elapsed_batch * 1000 / LOG_INTERVAL:5.2f} "
                  f"| loss {loss.item():5.3f} | ppl {math.exp(loss.item()):8.2f}")
            start_time = time.time() # リセット

    return total_loss / len(dataloader)

def main():
    print(f"Using device: {DEVICE}")

    # 1. トークナイザーのロード
    try:
        tokenizer = load_tokenizer(TOKENIZER_MODEL_PATH)
        SRC_VOCAB_SIZE = get_vocab_size(tokenizer)
        TGT_VOCAB_SIZE = SRC_VOCAB_SIZE # 四則演算なのでソースとターゲットで語彙を共有
        print(f"Tokenizer loaded. Vocabulary size: {SRC_VOCAB_SIZE}")
    except Exception as e:
        print(f"Failed to load tokenizer: {e}")
        return

    # 2. データセットとデータローダーの準備
    try:
        train_dataset = ArithmeticDataset(DATASET_PATH, tokenizer, max_length=MAX_SEQ_LEN_DATA)
        # train_dataset, val_dataset に分割することも推奨 (今回は省略)
        train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                                      shuffle=True, collate_fn=lambda b: collate_fn(b, PAD_ID))
        print(f"Dataset loaded. Training samples: {len(train_dataset)}")
    except FileNotFoundError:
        print(f"Error: Dataset file '{DATASET_PATH}' not found.")
        return
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # 3. モデルの初期化
    model = Seq2SeqTransformer(num_encoder_layers=NUM_ENCODER_LAYERS,
                               num_decoder_layers=NUM_DECODER_LAYERS,
                               emb_size=EMB_SIZE,
                               nhead=NHEAD,
                               src_vocab_size=SRC_VOCAB_SIZE,
                               tgt_vocab_size=TGT_VOCAB_SIZE,
                               dim_feedforward=FFN_HID_DIM,
                               dropout=DROPOUT,
                               max_seq_len=MAX_SEQ_LEN_MODEL
                              ).to(DEVICE)
    print(f"Model initialized with {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters.")


    # 4. 損失関数とオプティマイザ
    # PAD_ID を無視するように設定
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.98), eps=1e-9)
    # (オプション) 学習率スケジューラ
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1.0, gamma=0.95)


    # チェックポイントディレクトリ作成
    if not os.path.exists(CHECKPOINT_DIR):
        os.makedirs(CHECKPOINT_DIR)

    # --- 学習ループ ---
    print("\n--- Starting Training ---")
    for epoch in range(NUM_EPOCHS):
        epoch_start_time = time.time()
        
        avg_train_loss = train_epoch(model, train_dataloader, optimizer, criterion, TGT_VOCAB_SIZE, epoch)
        
        # (オプション) 検証ステップ
        # avg_val_loss = evaluate(model, val_dataloader, criterion, TGT_VOCAB_SIZE)
        
        epoch_duration = time.time() - epoch_start_time
        
        print("-" * 50)
        print(f"| End of Epoch {epoch+1:3d} | Time: {epoch_duration:5.2f}s | Avg Train Loss: {avg_train_loss:5.3f} | Avg Train PPL: {math.exp(avg_train_loss):8.2f}")
        # print(f"| Val Loss: {avg_val_loss:5.3f} | Val PPL: {math.exp(avg_val_loss):8.2f}") # 検証時
        print("-" * 50)

        # (オプション) 学習率スケジューラの更新
        # scheduler.step()

        # チェックポイントの保存
        if (epoch + 1) % CHECKPOINT_SAVE_INTERVAL == 0:
            checkpoint_path = os.path.join(CHECKPOINT_DIR, f"model_epoch_{epoch+1}.pt")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_train_loss,
                # ハイパーパラメータも保存しておくと便利
                'params': {
                    'emb_size': EMB_SIZE, 'nhead': NHEAD, 'ffn_hid_dim': FFN_HID_DIM,
                    'num_encoder_layers': NUM_ENCODER_LAYERS, 'num_decoder_layers': NUM_DECODER_LAYERS,
                    'dropout': DROPOUT, 'src_vocab_size': SRC_VOCAB_SIZE, 'tgt_vocab_size': TGT_VOCAB_SIZE,
                    'max_seq_len_model': MAX_SEQ_LEN_MODEL
                }
            }, checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")

    print("--- Training Finished ---")

if __name__ == "__main__":
    # 実行前に、TOKENIZER_MODEL_PATH と DATASET_PATH が正しいか確認してください。
    # また、ダミーのトークナイザーモデルとデータセットでテストする場合は、
    # それぞれのファイル (tokenizer_setup.py, dataset.py) の if __name__ == "__main__": ブロックで
    # ダミーファイルが作成されるようになっているか確認してください。
    
    # 例: ダミーデータでテストする場合
    # 1. tokenizer_setup.py を実行して arithmetic_tokenizer.model を作成 (コメントアウトを解除して実行)
    # 2. dataset.py を実行して dummy_dataset.jsonl を作成 (そのままでOK)
    # 3. この train.py を実行 (DATASET_PATH を "dummy_dataset.jsonl" に変更)
    
    main()