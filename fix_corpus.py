import os
from tqdm import tqdm

# --- 設定 ---
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))

# 入力ファイル名（現在、2行で1組になっているファイル）
INPUT_CORPUS_FILE_NAME = "corpus_for_tokenizer.txt" # これは以前のスクリプトで生成されたファイル名
INPUT_CORPUS_PATH = os.path.join(PROJECT_ROOT, INPUT_CORPUS_FILE_NAME)

# 出力ファイル名（整形後、1行にまとまったファイル）
# ★ここを修正します★
OUTPUT_FIXED_CORPUS_FILE_NAME = "math_tokenizer_corpus_expr_answer.txt"
OUTPUT_FIXED_CORPUS_PATH = os.path.join(PROJECT_ROOT, OUTPUT_FIXED_CORPUS_FILE_NAME)

# --- 処理関数 ---
def fix_corpus_format(input_path, output_path):
    print(f"Reading from '{input_path}' and fixing format...")
    
    combined_count = 0
    
    with open(input_path, 'r', encoding='utf-8') as infile, \
         open(output_path, 'w', encoding='utf-8', newline='') as outfile:
        
        lines = infile.readlines()
        
        if len(lines) % 2 != 0:
            print("Warning: The number of lines is odd. The last line might be incomplete or malformed.")
        
        for i in tqdm(range(0, len(lines), 2), desc="Combining lines"):
            if i + 1 < len(lines):
                expr_line = lines[i].strip()
                answer_line = lines[i+1].strip()
                
                combined_line = f"{expr_line} = {answer_line}"
                outfile.write(combined_line + "\n")
                combined_count += 1
            else:
                print(f"Skipping incomplete last line: '{lines[i].strip()}'")

    print(f"\nFormat fixing complete!")
    print(f"Combined {combined_count} pairs of lines.")
    print(f"Output saved to: {output_path}")

# --- スクリプト実行 ---
if __name__ == "__main__":
    fix_corpus_format(INPUT_CORPUS_PATH, OUTPUT_FIXED_CORPUS_PATH)