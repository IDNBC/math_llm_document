import sentencepiece as spm

spm.SentencePieceTrainer.train(
    input='input.txt',
    model_prefix='char_model',
    vocab_size=500,
    model_type='char',  # 文字単位で分割（初期評価用）
    character_coverage=1.0
)
