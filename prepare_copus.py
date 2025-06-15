# import json
# import os
# from tqdm import tqdm

# # ... (設定部分は同じ) ...

# def prepare_tokenizer_corpus(input_path, output_path):
#     print(f"Loading data from {input_path} and preparing corpus...")
    
#     processed_count = 0
#     skipped_count = 0
    
#     with open(input_path, 'r', encoding='utf-8') as infile, \
#          open(output_path, 'w', encoding='utf-8', newline='') as outfile:
        
#         for line_num, raw_line in enumerate(tqdm(infile, desc="Processing lines")):
#             try:
#                 # 行から不必要なホワイトスペース（改行含む）をすべて削除してからパース
#                 # この .strip() が行末のゴミを除去するはず
#                 cleaned_line = raw_line.strip() 
#                 if not cleaned_line: # 空行はスキップ
#                     skipped_count += 1
#                     continue

#                 item = json.loads(cleaned_line)
                
#                 expr = item.get("expr")
#                 answer = item.get("answer")
                
#                 if expr is not None and answer is not None:
#                     formatted_answer = str(answer) 
                    
#                     # f-stringでの結合は改行を含まない文字列を生成
#                     corpus_line = f"{expr} = {formatted_answer}"
#                     outfile.write(corpus_line + "\n") # ここで明示的に改行を追加
#                     processed_count += 1
#                 else:
#                     skipped_count += 1
#             except json.JSONDecodeError as e:
#                 print(f"Skipping malformed JSON line {line_num + 1}: {raw_line.strip()} - Error: {e}")
#                 skipped_count += 1
#             except Exception as e:
#                 print(f"An unexpected error occurred for line {line_num + 1}: {raw_line.strip()} - Error: {e}")
#                 skipped_count += 1

#     print(f"\nCorpus preparation complete!")
#     print(f"Processed {processed_count} valid entries.")
#     print(f"Skipped {skipped_count} invalid or malformed entries.")
#     print(f"Output corpus saved to: {output_path}")

# ... (スクリプト実行部分は同じ) ...


import json
import os
from tqdm import tqdm

def prepare_tokenizer_corpus(input_path, output_path):
    print(f"Loading data from {input_path} and preparing corpus...")

    processed_count = 0
    skipped_count = 0

    with open(input_path, 'r', encoding='utf-8') as infile, \
         open(output_path, 'w', encoding='utf-8', newline='') as outfile:

        for line_num, raw_line in enumerate(tqdm(infile, desc="Processing lines")):
            try:
                cleaned_line = raw_line.strip()
                if not cleaned_line:
                    skipped_count += 1
                    continue

                item = json.loads(cleaned_line)

                expr = item.get("expr")
                answer = item.get("answer")

                # 必要なのは expr と answer の両方が存在すること
                if expr is not None and answer is not None:
                    if isinstance(answer, (int, float, str)):
                    # 改行文字を削除または空白に変換
                        formatted_answer = str(answer).replace('\n', ' ').replace('\r', ' ')
                        corpus_line = f"{expr} = {formatted_answer}"
                        outfile.write(corpus_line + "\n")
                        processed_count += 1
                    else:
                        skipped_count += 1

                else:
                    skipped_count += 1
            except json.JSONDecodeError as e:
                print(f"Skipping malformed JSON line {line_num + 1}: {raw_line.strip()} - Error: {e}")
                skipped_count += 1
            except Exception as e:
                print(f"Unexpected error at line {line_num + 1}: {e}")
                skipped_count += 1

    print(f"\nCorpus preparation complete!")
    print(f"Processed {processed_count} valid entries.")
    print(f"Skipped {skipped_count} invalid or malformed entries.")
    print(f"Output corpus saved to: {output_path}")
