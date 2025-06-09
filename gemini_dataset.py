import json
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from gemini_tokenizer_setup import load_tokenizer, SOS_ID, EOS_ID, PAD_ID, TOKENIZER_MODEL_PATH
import torch # pad_sequenceのためにtorchをインポート

class ArithmeticDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length=50): # max_length を追加
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pairs = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                # "question" と "answer" キーが存在することを想定
                self.pairs.append((data['question'], data['answer']))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        question, answer = self.pairs[idx]

        # トークナイズ
        # エンコーダー入力: <sos> question <eos> (多くの実装では<sos>は不要、<eos>のみで区切る)
        # ここではシンプルに question をそのままエンコードします
        src_tokens = self.tokenizer.encode_as_ids(str(question))


        # デコーダー入力: <sos> answer
        # 損失計算用ターゲット: answer <eos>
        # sentencepieceのencode_as_idsはデフォルトでbos/eosを付与しないことが多い。
        # encode(..., add_bos=True, add_eos=True) のようにオプションで指定するか、手動で追加。
        # ここでは手動で追加します。
        tgt_tokens_input = [SOS_ID] + self.tokenizer.encode_as_ids(str(answer))
        tgt_tokens_label = self.tokenizer.encode_as_ids(str(answer)) + [EOS_ID]


        # 最大長を超える場合は切り詰める (省略する場合やエラーにする場合もあり)
        src_tokens = src_tokens[:self.max_length-1] # EOS用に1つ余裕を持たせる (エンコーダー入力にはEOS不要の場合も)
        tgt_tokens_input = tgt_tokens_input[:self.max_length]
        tgt_tokens_label = tgt_tokens_label[:self.max_length]


        return {
            "src": torch.tensor(src_tokens, dtype=torch.long),
            "tgt_input": torch.tensor(tgt_tokens_input, dtype=torch.long), # デコーダーへの入力
            "tgt_label": torch.tensor(tgt_tokens_label, dtype=torch.long)  # 損失計算時のターゲット
        }

def collate_fn(batch, pad_id=PAD_ID):
    """
    バッチ内のシーケンスをパディングしてテンソルに変換する関数
    """
    src_batch = [item['src'] for item in batch]
    tgt_input_batch = [item['tgt_input'] for item in batch]
    tgt_label_batch = [item['tgt_label'] for item in batch]

    src_padded = pad_sequence(src_batch, batch_first=True, padding_value=pad_id)
    tgt_input_padded = pad_sequence(tgt_input_batch, batch_first=True, padding_value=pad_id)
    tgt_label_padded = pad_sequence(tgt_label_batch, batch_first=True, padding_value=pad_id)
    
    # マスクも作成 (パディング部分を無視するため)
    # src_mask: (batch_size, src_len) PADの部分がTrue
    # tgt_mask: (batch_size, tgt_len) PADの部分がTrue
    # tgt_key_padding_mask: (batch_size, tgt_len) PADの部分がTrue (Transformerのデコーダ用)
    src_padding_mask = (src_padded == pad_id)
    tgt_padding_mask = (tgt_input_padded == pad_id) # デコーダー入力に対するパディングマスク

    return {
        "src": src_padded,
        "tgt_input": tgt_input_padded,
        "tgt_label": tgt_label_padded,
        "src_padding_mask": src_padding_mask,
        "tgt_padding_mask": tgt_padding_mask
    }


# --- テスト (このファイルを実行したときに動作確認) ---
if __name__ == "__main__":
    # ダミーのJSONLファイルを作成
    dummy_jsonl_path = "dummy_dataset.jsonl"
    with open(dummy_jsonl_path, 'w', encoding='utf-8') as f:
        f.write('{"question": "1+1", "answer": "2"}\n')
        f.write('{"question": "10-3", "answer": "7"}\n')
        f.write('{"question": "2*5", "answer": "10"}\n')
        f.write('{"question": "100/10", "answer": "10"}\n')
        f.write('{"question": "12+345*2", "answer": "702"}\n') # 長めの例

    try:
        tokenizer = load_tokenizer(TOKENIZER_MODEL_PATH) # 事前に学習済みのモデルが必要
        
        print("ArithmeticDataset のテスト:")
        dataset = ArithmeticDataset(dummy_jsonl_path, tokenizer, max_length=20) # max_lengthを適切に設定
        print(f"データセットのサイズ: {len(dataset)}")
        sample_data = dataset[0]
        print(f"最初のサンプルデータ: {sample_data}")
        print(f"  src (IDs): {sample_data['src'].tolist()}")
        print(f"  src (decoded): {tokenizer.decode_ids(sample_data['src'].tolist())}")
        print(f"  tgt_input (IDs): {sample_data['tgt_input'].tolist()}")
        print(f"  tgt_input (decoded): {tokenizer.decode_ids(sample_data['tgt_input'].tolist())}")
        print(f"  tgt_label (IDs): {sample_data['tgt_label'].tolist()}")
        print(f"  tgt_label (decoded): {tokenizer.decode_ids(sample_data['tgt_label'].tolist())}")

        print("\nDataLoader (collate_fnあり) のテスト:")
        dataloader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=collate_fn)
        batch_data = next(iter(dataloader))
        print(f"バッチデータのキー: {batch_data.keys()}")
        print(f"  src shape: {batch_data['src'].shape}")
        print(f"  src_padding_mask shape: {batch_data['src_padding_mask'].shape}")
        print(f"  tgt_input shape: {batch_data['tgt_input'].shape}")
        print(f"  tgt_padding_mask shape: {batch_data['tgt_padding_mask'].shape}")
        print(f"  tgt_label shape: {batch_data['tgt_label'].shape}")
        
        print("\nバッチデータの内容 (src):")
        print(batch_data['src'])
        print("バッチデータの内容 (tgt_input):")
        print(batch_data['tgt_input'])
        print("バッチデータの内容 (tgt_label):")
        print(batch_data['tgt_label'])


    except FileNotFoundError:
        print(f"エラー: トークナイザーモデル '{TOKENIZER_MODEL_PATH}' またはデータセットファイルが見つかりません。")
    except Exception as e:
        print(f"データローダーのテスト中にエラーが発生しました: {e}")