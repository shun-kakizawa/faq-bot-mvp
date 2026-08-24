"""APIキー不要のパイプライン検証テスト。

OpenAI の代わりに MockEmbedding / MockLLM を使い、
「文書取り込み → ベクトル化 → 検索 → 回答生成」という
RAG の一連の流れがネットワーク接続なしで最後まで通ることを確認する。

これにより、OpenAI のコストをかけずに、コードの配線（構造）が
壊れていないことを CI 等でも自動チェックできる。

実行:
    pytest -q
または
    python tests/test_pipeline.py
"""
import os
import sys
import tempfile

# リポジトリ直下を import パスに追加（pytest / 直接実行の両対応）
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llama_index.core import (  # noqa: E402
    SimpleDirectoryReader,
    VectorStoreIndex,
    Settings,
)
from llama_index.core.llms import MockLLM  # noqa: E402
from llama_index.core import MockEmbedding  # noqa: E402
from llama_index.core.node_parser import SentenceSplitter  # noqa: E402

from src import config  # noqa: E402


def build_test_engine():
    """モックのモデルを使い、data/ からインメモリでインデックスを構築して
    クエリエンジンを返す。Chroma には触れず、テストを軽量・独立に保つ。
    """
    # OpenAI の代わりにモックを全体設定へ差し込む
    Settings.embed_model = MockEmbedding(embed_dim=256)
    Settings.llm = MockLLM(max_tokens=64)

    documents = SimpleDirectoryReader(config.DATA_DIR).load_data()
    splitter = SentenceSplitter(
        chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP
    )
    index = VectorStoreIndex.from_documents(documents, transformations=[splitter])
    return index.as_query_engine(similarity_top_k=config.SIMILARITY_TOP_K)


def test_documents_are_loaded():
    """data/ に文書が存在し、読み込めることを確認。"""
    documents = SimpleDirectoryReader(config.DATA_DIR).load_data()
    assert len(documents) > 0, "data/ から文書を1件も読み込めていません"


def test_pipeline_runs_and_retrieves_sources():
    """検索→回答生成が例外なく通り、根拠文書が取得できることを確認。"""
    engine = build_test_engine()
    response = engine.query("有給休暇は入社何か月で付与されますか？")

    # 回答テキストが返ってくること（モックなので内容は問わない）
    assert str(response).strip() != ""
    # 検索で関連チャンク（根拠）が拾えていること
    assert len(response.source_nodes) > 0, "関連文書が検索できていません"


if __name__ == "__main__":
    # pytest が無い環境でも動くよう、手動実行に対応
    test_documents_are_loaded()
    test_pipeline_runs_and_retrieves_sources()
    print("OK: すべてのチェックを通過しました（モックによるパイプライン検証成功）")
