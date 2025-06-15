import streamlit as st
from pyaskit import TypedDict, List
from pyaskit import function

# TypedDictでクイズ形式を定義
class Quiz(TypedDict):
    question: str
    choices: List[str]
    model_answer: int

@function(codable=False)
def make_quiz(category: str, n: int, count: int, difficulty: int) -> List[Quiz]:
    """{{category}}分野から{{count}}個の{{n}}択問題(question)、
    選択肢(choices)、模範解答(model_answer)を日本語で作成してください。
    模範解答の選択肢の番号は0から{{n}}-1とします。
    難易度は{{difficulty}}とします。1が最も簡単で10が最も難しくなります。"""

st.title("クイズアプリ")
# カテゴリ
category = st.text_input("カテゴリを入力してください:", value="プログラミング")
# 選択肢の数
n_choices = st.slider("選択肢の数:", min_value=3, max_value=5, value=4)
# 難易度
difficulty = st.slider("難易度:", min_value=1, max_value=10, value=5)
# 問題数
question_count = st.slider("問題数:", min_value=1, max_value=10, value=5)

# セッション状態がクイズでないと初期化
if "quizzes" not in st.session_state:
    st.session_state.quizzes = []

# クイズ生成ボタン
if st.button("クイズを生成"):
    # クイズ生成でセッション状態に保存
    quizzes = make_quiz(category, n_choices, question_count, difficulty)
    st.session_state.quizzes = quizzes
    # ユーザの解答格納リスト初期化
    st.session_state.user_answers = [0] * len(quizzes)

# クイズと選択肢表示
for i, quiz in enumerate(st.session_state.quizzes):
    st.write(f"Q{i+1}: {quiz['question']}")
    options = quiz["choices"]
    # 選択肢のラジオボタン表示
    answer = st.radio("選択肢:", options, key=f"question_{i}")
    # ユーザの解答を更新する。
    st.session_state.user_answers[i] = options.index(answer)

# 採点するボタン
if st.button("採点"):
    score = 0
    # 正解数を数える
    for i, quiz in enumerate(st.session_state.quizzes):
        if quiz["model_answer"] == st.session_state.user_answers[i]:
            score += 1
    # スコア表示
    st.write(f"スコア: {score}/{len(st.session_state.quizzes)}")
    # 正解を表示
    st.write("正解")
    for i, quiz in enumerate(st.session_state.quizzes):
        st.write(f"Q{i+1}: {quiz['choices'][quiz['model_answer']]}")
