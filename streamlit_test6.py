# python -m streamlit run streamlit_test3.py
import streamlit as st #短い名前を付けるのは慣例
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbedding
from operator import itemgetter
from langchain_core.output_parsers import StrOutputParser
import base64

# 画像の説明を取得
def get_image_description(image_data: str):
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "human",
                [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_data}"},
                    }
                ],
            ),
        ]
    )
    chain = prompt | ChatOpenAI(model="gpt-4o-mini") | StrOutputParser()
    return chain.invoke({"image_data": image_data})

# メッセージ作成
def create_message(dic: dict):
    image_data = dic["image"]
    if image_data:
        return [
            (
                "human",
                [
                    {"type": "text", "text": dic["input"]},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_data}"},
                    },
                ],
            )
        ]
    return [("human", dic["input"])]

# ドキュメント整形
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# チェーンを作成
def create_chain():
    vectorstore = Chroma(
        embedding_function=OpenAIEmbeddings(model="text-embedding-3-small"),
        persist_directory="data",
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "回答には以下の情報も参考にしてください。参考情報：\n{info}",
            ),
            ("placeholder", "{history}"),
            ("placeholder", "{message}"),
        ]
    )
    return (
        {
            "message": create_message,
            "info": itemgetter("input") | retriever | format_docs,
            "history": itemgetter("history"),
        }
        | prompt
        | ChatOpenAI(model="gpt-4o-mini", temperature=0)
    )


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
    image_data = None
    image_description = ""
    if uploaded_file is not None:
        image_data = base64.b64encode(uploaded_file.read()).decode("utf-8")
        image_description = get_image_description(image_data)
    response = st.session_state.chain.invoke(
        {
            "input": user_input + image_description,
            "history": st.session_state.history,
            "image": image_data,
        }
    )
    st.session_state.history.append(HumanMessage(user_input))
    st.session_state.history.append(response)

    # 会話の表示
    for message in reversed(st.session_state.history):
        st.write(f"{message.type}: {message.content}")

