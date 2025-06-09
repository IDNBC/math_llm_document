import torch
import torch.nn as nn
import math
from tokenizer_setup import PAD_ID # PAD_IDをインポート

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # (1, max_len, d_model)
        self.register_buffer('pe', pe) # bufferとして登録 (モデルパラメータではない)

    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, d_model)
        """
        # x のシーケンス長に合わせて pe をスライス
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class Seq2SeqTransformer(nn.Module):
    def __init__(self,
                 num_encoder_layers: int,
                 num_decoder_layers: int,
                 emb_size: int, # d_model と同じ
                 nhead: int,    # MultiheadAttentionのヘッド数
                 src_vocab_size: int,
                 tgt_vocab_size: int,
                 dim_feedforward: int = 512, # FFNの隠れ層の次元
                 dropout: float = 0.1,
                 max_seq_len: int = 50 # PositionalEncodingのmax_len用
                ):
        super(Seq2SeqTransformer, self).__init__()
        
        self.emb_size = emb_size
        self.src_tok_emb = nn.Embedding(src_vocab_size, emb_size)
        self.tgt_tok_emb = nn.Embedding(tgt_vocab_size, emb_size)
        self.positional_encoding = PositionalEncoding(emb_size, dropout=dropout, max_len=max_seq_len)

        # PyTorch の Transformer モジュールを使用
        # batch_first=True を指定することが重要
        self.transformer = nn.Transformer(d_model=emb_size,
                                          nhead=nhead,
                                          num_encoder_layers=num_encoder_layers,
                                          num_decoder_layers=num_decoder_layers,
                                          dim_feedforward=dim_feedforward,
                                          dropout=dropout,
                                          batch_first=True) # batch_first=True を指定

        self.generator = nn.Linear(emb_size, tgt_vocab_size) # 出力層

        # Xavier Glorot 初期化 (任意)
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _generate_square_subsequent_mask(self, sz, device):
        """
        デコーダーの自己アテンション層で未来のトークンを見ないようにするためのマスクを生成
        mask[i, j] = True の場合、その位置は無視される
        sz: ターゲットシーケンス長
        """
        mask = (torch.triu(torch.ones(sz, sz, device=device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self,
                src: torch.Tensor, # (batch_size, src_seq_len)
                tgt_input: torch.Tensor, # (batch_size, tgt_seq_len) デコーダー入力
                src_padding_mask: torch.Tensor, # (batch_size, src_seq_len) Trueの部分がPAD
                tgt_padding_mask: torch.Tensor, # (batch_size, tgt_seq_len) Trueの部分がPAD
                memory_key_padding_mask: torch.Tensor # (batch_size, src_seq_len) エンコーダー出力のPADマスク
               ):
        """
        src_padding_mask: エンコーダーの Self-Attention と、エンコーダー出力をデコーダーが参照する際の Cross-Attention で使用
        tgt_padding_mask: デコーダーの Self-Attention で使用
        memory_key_padding_mask: デコーダーの Cross-Attention でエンコーダー出力のどの部分を無視するかを指定 (src_padding_maskと同じで良い)
        """
        device = src.device
        
        # --- エンコーダー ---
        src_emb = self.positional_encoding(self.src_tok_emb(src) * math.sqrt(self.emb_size))
        # src_emb shape: (batch_size, src_seq_len, emb_size)
        memory = self.transformer.encoder(src_emb, src_key_padding_mask=src_padding_mask)
        # memory shape: (batch_size, src_seq_len, emb_size)

        # --- デコーダー ---
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(tgt_input) * math.sqrt(self.emb_size))
        # tgt_emb shape: (batch_size, tgt_seq_len, emb_size)
        
        # デコーダーの未来マスクを作成
        tgt_seq_len = tgt_input.size(1)
        tgt_mask = self._generate_square_subsequent_mask(tgt_seq_len, device) # (tgt_seq_len, tgt_seq_len)

        # デコーダー実行
        # tgt_mask: 未来の情報を隠すためのマスク
        # tgt_key_padding_mask: ターゲットシーケンス自体のパディングを隠すマスク
        # memory_key_padding_mask: エンコーダー出力のパディングを隠すマスク (src_padding_mask を使う)
        decoder_output = self.transformer.decoder(tgt_emb, memory,
                                                  tgt_mask=tgt_mask,
                                                  tgt_key_padding_mask=tgt_padding_mask,
                                                  memory_key_padding_mask=memory_key_padding_mask) # src_padding_maskを使用
        # decoder_output shape: (batch_size, tgt_seq_len, emb_size)

        # --- 出力層 ---
        logits = self.generator(decoder_output)
        # logits shape: (batch_size, tgt_seq_len, tgt_vocab_size)
        return logits

# --- テスト (このファイルを実行したときに動作確認) ---
if __name__ == "__main__":
    from tokenizer_setup import load_tokenizer, get_vocab_size, TOKENIZER_MODEL_PATH
    
    # デバイス設定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    try:
        tokenizer = load_tokenizer(TOKENIZER_MODEL_PATH)
        VOCAB_SIZE = get_vocab_size(tokenizer)
        print(f"Tokenizer loaded. Vocab size: {VOCAB_SIZE}")

        # モデルパラメータ (軽量を目指すので小さめに)
        EMB_SIZE = 128        # 埋め込み次元 (d_model)
        NHEAD = 4             # マルチヘッドアテンションのヘッド数
        FFN_HID_DIM = 256     # FFNの隠れ層の次元
        NUM_ENCODER_LAYERS = 2
        NUM_DECODER_LAYERS = 2
        DROPOUT = 0.1
        MAX_SEQ_LEN_MODEL = 50 # データセットの最大長に合わせて調整

        model = Seq2SeqTransformer(num_encoder_layers=NUM_ENCODER_LAYERS,
                                   num_decoder_layers=NUM_DECODER_LAYERS,
                                   emb_size=EMB_SIZE,
                                   nhead=NHEAD,
                                   src_vocab_size=VOCAB_SIZE, # 簡単のためsrc/tgtで語彙を共有
                                   tgt_vocab_size=VOCAB_SIZE,
                                   dim_feedforward=FFN_HID_DIM,
                                   dropout=DROPOUT,
                                   max_seq_len=MAX_SEQ_LEN_MODEL
                                  ).to(device)

        print(f"Model initialized with {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters.")

        # ダミー入力データでフォワードパスをテスト
        batch_size = 2
        src_seq_len = 10
        tgt_seq_len = 8 # デコーダー入力の長さ

        # (batch_size, seq_len)
        dummy_src = torch.randint(0, VOCAB_SIZE, (batch_size, src_seq_len), device=device)
        dummy_tgt_input = torch.randint(0, VOCAB_SIZE, (batch_size, tgt_seq_len), device=device)
        
        # パディングマスク (0: トークン, 1: パッド) -> Transformerの期待は True でマスク
        dummy_src_padding_mask = torch.zeros((batch_size, src_seq_len), dtype=torch.bool, device=device)
        dummy_src_padding_mask[:, -2:] = True # 後ろ2つをパッドと仮定

        dummy_tgt_padding_mask = torch.zeros((batch_size, tgt_seq_len), dtype=torch.bool, device=device)
        dummy_tgt_padding_mask[:, -1:] = True # 後ろ1つをパッドと仮定
        
        print(f"Dummy src shape: {dummy_src.shape}")
        print(f"Dummy tgt_input shape: {dummy_tgt_input.shape}")
        print(f"Dummy src_padding_mask shape: {dummy_src_padding_mask.shape}")
        print(f"Dummy tgt_padding_mask shape: {dummy_tgt_padding_mask.shape}")
        
        # memory_key_padding_mask は src_padding_mask を使う
        output_logits = model(src=dummy_src,
                              tgt_input=dummy_tgt_input,
                              src_padding_mask=dummy_src_padding_mask,
                              tgt_padding_mask=dummy_tgt_padding_mask,
                              memory_key_padding_mask=dummy_src_padding_mask) # src_padding_maskを使用

        print(f"Output logits shape: {output_logits.shape}") # (batch_size, tgt_seq_len, tgt_vocab_size)
        assert output_logits.shape == (batch_size, tgt_seq_len, VOCAB_SIZE)
        print("Model forward pass test successful!")

    except FileNotFoundError:
        print(f"エラー: トークナイザーモデル '{TOKENIZER_MODEL_PATH}' が見つかりません。")
    except Exception as e:
        print(f"モデルのテスト中にエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()