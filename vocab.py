def find_token_in_vocab(token, vocab_path="math_tokenizer.vocab"):
    with open(vocab_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith(token + "\t"):
                print("Found:", line.strip())
                return
    print("Token not found.")

find_token_in_vocab("\\frac")
find_token_in_vocab("x")
find_token_in_vocab("▁x")
