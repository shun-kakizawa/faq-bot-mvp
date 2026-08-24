"""RAG の「回答」フェーズ。

Chroma に保存済みのインデックスを読み込み、質問に対して
関連文書を検索（Retrieval）→ その文書を根拠に LLM が回答を生成
（Generation）する。この2段構えが RAG（Retrieval-Augmented
Generation）の中核。
"""
from dataclasses import dataclass, field

from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.prompts import PromptTemplate
from llama_index.llms.openai import OpenAI

from . import config
from .store import get_vector_store


@dataclass
class Answer:
    """回答結果をまとめた入れ物。

    text   : 生成された回答文
    sources: 回答の根拠に使われた文書名の一覧（出典表示に使う）
    """
    text: str
    sources: list[str] = field(default_factory=list)


def configure_llm() -> None:
    """回答生成用の LLM（OpenAI）を全体設定に登録する。
    テスト時は呼ばず、MockLLM を Settings に差し込む。
    """
    Settings.llm = OpenAI(model=config.LLM_MODEL, temperature=0.1)


def load_query_engine(similarity_top_k: int | None = None):
    """Chroma の既存インデックスからクエリエンジンを構築する。

    from_vector_store を使うことで、再度埋め込みし直すことなく、
    保存済みのベクトルを使って検索できる。
    """
    vector_store = get_vector_store()
    index = VectorStoreIndex.from_vector_store(vector_store)

    return index.as_query_engine(
        similarity_top_k=similarity_top_k or config.SIMILARITY_TOP_K,
        # 根拠限定・ハルシネーション抑制のための日本語プロンプトを適用
        text_qa_template=PromptTemplate(config.QA_PROMPT_TEMPLATE),
    )


def answer(question: str, query_engine=None) -> Answer:
    """質問文を受け取り、根拠となる文書を検索したうえで回答を返す。"""
    engine = query_engine or load_query_engine()
    response = engine.query(question)

    # 回答の根拠に使われたチャンクの元ファイル名を重複なく集める
    sources: list[str] = []
    for node in getattr(response, "source_nodes", []):
        name = node.node.metadata.get("file_name")
        if name and name not in sources:
            sources.append(name)

    return Answer(text=str(response), sources=sources)
