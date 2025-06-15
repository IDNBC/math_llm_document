# python -m streamlit run streamlit_test3.py
import streamlit as st #短い名前を付けるのは慣例
from langchain_openai import ChatOpenAI
import langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate

def create_chain():
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "回答には以下の情報も参考にしてください。参考情報：\n{info}",
            ),
            ("placeholder", "{history}"),
            ("human", "{input}"),
        ]
    )
    return prompt | ChatOpenAI(model="gpt-4o-mini", temperature=0)

# セッションの初期化
if "history" not in st.session_state:
    st.session_state.history = []
    st.session_state.llm = ChatOpenAI()
    st.session_state.chain = create_chain()

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
    response = st.session_state.chain.invoke(
        {
            "input": user_input,
            "history": st.session_state.history,
            "info": "ユーザの年齢は10歳です。",
        }
    )
    st.session_state.history.append(HumanMessage(user_input))
    st.session_state.history.append(response)

    # 会話の表示
    for message in reversed(st.session_state.history):
        st.write(f"{message.type}: {message.content}")

