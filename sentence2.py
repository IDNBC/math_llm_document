import sentencepiece as spm

spm.SentencePieceTrainer.train(
    input="input.txt",
    model_prefix="math_tokenizer",
    vocab_size=100,
    model_type="unigram",  # あるいは 'bpe'
    character_coverage=1.0,
    user_defined_symbols=["\\frac", "\\sqrt", "\\int", "+", "-", "*", "(", ")", "="]
)
