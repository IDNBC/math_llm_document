from flask import Flask, render_template, jsonify, request

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/latex', methods=['GET'])
def get_latex():
    latex_expression = r'E=mc^2'
    return jsonify({'latex': latex_expression})

if __name__ == '__main__':
    app.run(debug=True)