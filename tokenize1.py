import sentencepiece as spm

sp = spm.SentencePieceProcessor()
sp.load("char_model.model")

text = r"\frac{a}{b} + \sqrt{x}"
tokens = sp.encode(text, out_type=str)
print(tokens)
