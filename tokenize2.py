import sentencepiece as spm

# モデル読み込み
sp = spm.SentencePieceProcessor()
sp.load("math_tokenizer.model")

# テストするLaTeX数式
latex_expr = r"\frac{a}{b} + \sqrt{x}"

# トークナイズ
tokens = sp.encode(latex_expr, out_type=str)
print("Tokenized:", tokens)

# ID列でも表示したい場合：
ids = sp.encode(latex_expr, out_type=int)
print("Token IDs:", ids)

# 元に戻す（デコード）
decoded = sp.decode(tokens)
print("Decoded:", decoded)
