import json

# 入力ファイル（例：formulas.json）
with open("linear.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# 出力ファイル
with open("linear.jsonl", "w", encoding="utf-8") as f_jsonl, \
     open("input.txt", "w", encoding="utf-8") as f_txt:

    for item in data:
        f_jsonl.write(json.dumps(item, ensure_ascii=False) + "\n")

        # LaTeX形式の式をトークナイザー用に書き出す
        if "latex_original" in item:
            f_txt.write(item["latex_original"] + "\n")
        if "latex_transformed" in item:
            f_txt.write(item["latex_transformed"] + "\n")
