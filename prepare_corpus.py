import json

JSONL_PATH = "data_set/dataset_all_combined3.jsonl"  # ご自身のデータセットのパス
CORPUS_PATH = "corpus_for_tokenizer.txt"     # 出力するテキストファイルのパス

print(f"'{JSONL_PATH}' からコーパスを作成します...")

with open(JSONL_PATH, 'r', encoding='utf-8') as infile, \
     open(CORPUS_PATH, 'w', encoding='utf-8') as outfile:
    
    line_count = 0
    for line in infile:
        try:
            data = json.loads(line)
            
            # 問題文 (expr) を書き出す
            if 'expr' in data:
                outfile.write(str(data['expr']) + '\n')
            
            # 解答 (answer) を書き出す
            # answer が None の場合は "None" という文字列として扱う
            if 'answer' in data:
                outfile.write(str(data['answer']) + '\n')
            
            line_count += 1
        except json.JSONDecodeError:
            print(f"警告: JSONのデコードに失敗した行があります: {line.strip()}")

print(f"完了しました。{line_count}行のデータから '{CORPUS_PATH}' を作成しました。")