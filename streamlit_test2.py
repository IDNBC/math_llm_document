# python -m streamlit run streamlit_test2.py
import streamlit as st #短い名前を付けるのは慣例
from langchain_openai import ChatOpenAI

# タイトルの設定
st.title('マルチモーダルRAGチャットボット')

# 画像をアップロードするウィジェット
uploaded_file = st.file_uploader("画像を選択してください", type=["jpg", "jpeg", "png"])

# 画像があればそれを表示する
if uploaded_file is not None:
    st.image(uploaded_file, caption="画像", width=300)

# テキスト入力フィールド
user_input = st.text_input("メッセージを入力してください：")



# 送信ボタンが押されたら動く
if st.button("送信"):
    # OpenAI応答
    llm = ChatOpenAI()
    response = llm.invoke(user_input)

    st.write(f"ai: {response.content}")
    st.write(f"human: {user_input}")
