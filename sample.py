import json
import re

with open("sample.jsonl", "r", encoding="utf-8") as infile, \
     open("output.txt", "w", encoding="utf-8") as outfile:

    for line in infile:
        item = json.loads(line.strip())
        expr = item.get("expr")
        answer = item.get("answer")
        if expr is not None and answer is not None:
            formatted_answer = re.sub(r'[\r\n]+', ' ', str(answer)).strip()
            corpus_line = f"{expr} = {formatted_answer}"
            print(f"DEBUG: {repr(corpus_line)}")
            outfile.write(corpus_line + "\n")
