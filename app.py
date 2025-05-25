# from flask import Flask, render_template, jsonify, request

# app = Flask(__name__)

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/latex', methods=['GET'])
# def get_latex():
#     latex_expression = r'E=mc^2'
#     return jsonify({'latex': latex_expression})

# if __name__ == '__main__':
#     app.run(debug=True)

# from flask import Flask, render_template, request, jsonify
# from transformers import AutoTokenizer, AutoModelForCausalLM
# import torch

# app = Flask(__name__)

# # 保存されたモデルとトークナイザーの読み込み
# tokenizer = AutoTokenizer.from_pretrained("./saved_model")
# model = AutoModelForCausalLM.from_pretrained("./saved_model")

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/ask', methods=['POST'])
# def ask():
#     user_input = request.json.get("question", "")

#     # 入力のトークナイズ
#     inputs = tokenizer.encode(user_input, return_tensors="pt")

#     # モデルによる出力生成
#     outputs = model.generate(inputs, max_new_tokens=20, pad_token_id=tokenizer.eos_token_id)

#     # デコードして返す
#     answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

#     return jsonify({'answer': answer})

# if __name__ == '__main__':
#     app.run(debug=True)

from flask import Flask, render_template, request, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Flask アプリ初期化
app = Flask(__name__)

# モデルとトークナイザーの読み込み
model_path = "./saved_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    data = request.get_json()
    user_input = data.get("message", "")

    if not user_input:
        return jsonify({'error': '空の入力です'}), 400

    try:
        input_ids = tokenizer.encode(user_input, return_tensors="pt")
        output_ids = model.generate(input_ids, max_new_tokens=20, pad_token_id=tokenizer.eos_token_id)
        output = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return jsonify({'response': output})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
