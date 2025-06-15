import json
import os
from tqdm import tqdm

# --- 設定 ---
# このスクリプトが格納されているディレクトリをプロジェクトルートとする
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))

# 入力データセットファイル名
INPUT_DATASET_FILE_NAME = "data_set/dataset_all_combined3.jsonl"
INPUT_DATASET_PATH = os.path.join(PROJECT_ROOT, INPUT_DATASET_FILE_NAME)

# 出力するモデル学習用データセットファイル名
# ここでは新しい名前を提案しますが、必要に応じて変更してください
OUTPUT_MODEL_DATASET_FILE_NAME = "math_model_data_expr_answer.jsonl"
OUTPUT_MODEL_DATASET_PATH = os.path.join(PROJECT_ROOT, OUTPUT_MODEL_DATASET_FILE_NAME)

# --- データ抽出と整形 ---
def prepare_model_dataset(input_path, output_path):
    print(f"Loading data from {input_path} and preparing model dataset...")
    
    processed_count = 0
    skipped_count = 0
    
    with open(input_path, 'r', encoding='utf-8') as infile, \
         open(output_path, 'w', encoding='utf-8', newline='') as outfile:
        
        for line_num, raw_line in enumerate(tqdm(infile, desc="Processing lines")):
            try:
                cleaned_line = raw_line.strip() 
                if not cleaned_line: # 空行はスキップ
                    skipped_count += 1
                    continue

                item = json.loads(cleaned_line)
                
                expr = item.get("expr")
                answer = item.get("answer")
                
                # exprとanswerが両方存在することを確認
                if expr is not None and answer is not None:
                    # 新しいJSONオブジェクトを生成し、必要なキーのみを含める
                    new_item = {
                        "expr": expr,
                        "answer": answer
                    }
                    # 新しいJSONオブジェクトを1行のJSONL形式で出力
                    outfile.write(json.dumps(new_item, ensure_ascii=False) + "\n")
                    processed_count += 1
                else:
                    skipped_count += 1
                    print(f"Skipping line {line_num + 1} due to missing 'expr' or 'answer': {raw_line.strip()}")
            except json.JSONDecodeError as e:
                print(f"Skipping malformed JSON line {line_num + 1}: {raw_line.strip()} - Error: {e}")
                skipped_count += 1
            except Exception as e:
                print(f"An unexpected error occurred for line {line_num + 1}: {raw_line.strip()} - Error: {e}")
                skipped_count += 1

    print(f"\nModel dataset preparation complete!")
    print(f"Processed {processed_count} valid entries.")
    print(f"Skipped {skipped_count} invalid or malformed entries.")
    print(f"Output model dataset saved to: {output_path}")

# --- スクリプト実行 ---
if __name__ == "__main__":
    prepare_model_dataset(INPUT_DATASET_PATH, OUTPUT_MODEL_DATASET_PATH)