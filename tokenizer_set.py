import sentencepiece as spm
import torch

# --- トークナイザー関連の定数 ---
PAD_TOKEN = "<pad>"
SOS_TOKEN = "<sos>"
EOS_TOKEN = "<eos>"
UNK_TOKEN = "<unk>"

PAD_ID = 0
SOS_ID = 1
EOS_ID = 2
UNK_ID = 3

# SentencePieceモデルのパス (事前に学習させておく必要があります)
TOKENIZER_MODEL_PATH = "arithmetic_tokenizer.model" # ご自身のモデルパスに変更してください

def load_tokenizer(model_path=TOKENIZER_MODEL_PATH):
    """SentencePieceモデルをロードする関数"""
    sp = spm.SentencePieceProcessor()
    sp.load(model_path)
    # SentencePieceのデフォルトのIDと整合性を取るために、特殊トークンのIDを上書き
    # モデル学習時に user_defined_symbols で指定している場合は不要な場合もあります
    sp.SetEncodeExtraOptions("bos:eos") # これにより <sos> と <eos> が自動で付与されるようになる場合があるが、明示的に扱う方が確実
    
    # SentencePieceの語彙に特殊トークンが含まれているか確認し、IDを取得
    # 含まれていない場合はエラーとするか、動的に追加する処理が必要
    # ここでは、学習済みモデルに特殊トークンが登録されている前提とします。
    # 例: sp.piece_to_id(PAD_TOKEN) などでIDを取得
    
    # 今回は簡単のため、SentencePieceのデフォルトIDを使いつつ、
    # PAD_IDは別途扱う想定で進めます。
    # 実際の語彙サイズは sp.get_piece_size() で取得できます。
    return sp

def get_vocab_size(sp_processor):
    """トークナイザーの語彙サイズを取得する関数"""
    return sp_processor.get_piece_size()

# --- テスト (このファイルを実行したときに動作確認) ---
if __name__ == "__main__":
    # SentencePieceモデルを学習させる例 (初回のみ、または語彙を変更する場合)
    # この部分は通常、別途スクリプト (例: 00_train_tokenizer.py) で実行します
    # ここでは、モデルが既に存在することを前提とします。
    # もしモデルがない場合は、以下の様なコードで学習できます。
    # import glob
    # files = glob.glob("path_to_your_text_data/*.txt") # データセットからテキストデータを抽出してファイルに保存
    # spm.SentencePieceTrainer.Train(
    #     input=','.join(files),
    #     model_prefix='arithmetic_tokenizer', # 出力モデル名
    #     vocab_size=100, # 四則演算なので小さめでOK (例: 0-9, +, -, *, /, =, ., <sos>, <eos>, <pad>, <unk>)
    #     user_defined_symbols=[PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN],
    #     pad_id=PAD_ID,
    #     bos_id=SOS_ID, # SentencePieceではbos_idが<sos>に相当
    #     eos_id=EOS_ID, # SentencePieceではeos_idが<eos>に相当
    #     unk_id=UNK_ID,
    #     model_type='bpe', # または 'unigram'
    #     character_coverage=1.0 # 数字や演算子をカバー
    # )
    # print(f"'{TOKENIZER_MODEL_PATH}' を学習しました。")
    
    try:
        tokenizer = load_tokenizer()
        print(f"トークナイザーをロードしました。語彙サイズ: {get_vocab_size(tokenizer)}")
        
        # トークナイズのテスト
        test_text = "123+45="
        encoded_ids = tokenizer.encode_as_ids(test_text)
        print(f"'{test_text}' -> IDs: {encoded_ids}")
        decoded_text = tokenizer.decode_ids(encoded_ids)
        print(f"IDs -> '{decoded_text}'")

        # 特殊トークンのID確認 (SentencePieceの仕様により、学習時の設定に依存します)
        print(f"PAD ID (SentencePiece default): {tokenizer.pad_id()}") # 通常 0 or -1 (未指定時)
        print(f"SOS ID (SentencePiece default): {tokenizer.bos_id()}") # 通常 1
        print(f"EOS ID (SentencePiece default): {tokenizer.eos_id()}") # 通常 2
        print(f"UNK ID (SentencePiece default): {tokenizer.unk_id()}") # 通常 0 or 3 (モデルによる)

        # グローバル定数との比較
        print(f"Defined PAD_ID: {PAD_ID}")
        print(f"Defined SOS_ID: {SOS_ID}")
        print(f"Defined EOS_ID: {EOS_ID}")
        
        # 注意: SentencePieceの `pad_id()` はデフォルトで0を返すことが多いですが、
        # `user_defined_symbols` で指定した際の実際のIDと異なる場合があります。
        # `model_proto.trainer_spec.pad_id` や `model_proto.normalizer_spec. Getränkepfad_id` を確認するか、
        # `piece_to_id` で明示的に確認することを推奨します。
        # ここでは、PAD_ID = 0 を前提として collate_fn で使用します。

    except Exception as e:
        print(f"エラー: {e}")
        print(f"SentencePieceモデル '{TOKENIZER_MODEL_PATH}' が見つからないか、ロードに失敗しました。")
        print("事前にトークナイザーモデルを学習し、正しいパスを指定してください。")