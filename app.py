"""Streamlit 製のチャットUI（アプリのエントリポイント）。

実行方法:
    1. 事前に `python build_index.py` でインデックスを構築
    2. `streamlit run app.py` で起動

ブラウザ上のチャット画面から社内FAQに質問でき、
回答とその根拠になった文書名（出典）が表示される。
"""
import os

import streamlit as st

from src import config
from src.rag import configure_llm, load_query_engine, answer


st.set_page_config(page_title="社内FAQアシスタント", page_icon="💬")
st.title("💬 社内FAQアシスタント")
st.caption("人材派遣会社の社内問い合わせ（有給・経費・就業規則など）に、社内文書をもとに答えます。")


@st.cache_resource(show_spinner="インデックスを読み込んでいます...")
def get_engine():
    """クエリエンジンを一度だけ構築してキャッシュする。

    st.cache_resource により、質問のたびに再読み込みせず
    高速に応答できる。
    """
    configure_llm()
    return load_query_engine()


def main() -> None:
    # APIキー未設定なら、動かす前に分かりやすく案内する
    if not os.getenv("OPENAI_API_KEY"):
        st.warning(
            "OpenAI APIキーが設定されていません。プロジェクト直下の `.env` に "
            "`OPENAI_API_KEY=...` を設定してから再起動してください。"
        )
        st.stop()

    # Chroma にデータが無い（インデックス未構築）場合の案内
    if not os.path.isdir(config.CHROMA_DIR):
        st.warning(
            "インデックスがまだありません。ターミナルで `python build_index.py` を "
            "実行してから、この画面を再読み込みしてください。"
        )
        st.stop()

    engine = get_engine()

    # これまでのやり取りをセッションに保持
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # 履歴の再表示
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # 入力欄
    if prompt := st.chat_input("質問を入力してください（例：有給は入社何か月で付与されますか？）"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("社内文書を検索して回答を作成しています..."):
                result = answer(prompt, query_engine=engine)

            content = result.text
            if result.sources:
                content += "\n\n---\n**参照した社内文書:** " + ", ".join(result.sources)
            st.markdown(content)

        st.session_state.messages.append({"role": "assistant", "content": content})


if __name__ == "__main__":
    main()
