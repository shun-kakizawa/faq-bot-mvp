"""インジェスト（取り込み）処理。

data/ フォルダの文書を読み込み → 適切なサイズに分割 →
埋め込みベクトルに変換 → Chroma に保存する。
RAG の「準備（インデックス構築）」フェーズを担当する。

コマンドラインからは `python build_index.py` で実行する。
"""
from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    StorageContext,
    Settings,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.openai import OpenAIEmbedding

from . import config
from .store import get_vector_store


def configure_embeddings() -> None:
    """埋め込みモデル（OpenAI）を LlamaIndex の全体設定に登録する。

    テスト時はこの関数を呼ばず、MockEmbedding を Settings に
    差し込むことで、APIキー無しでもパイプラインを検証できる。
    """
    Settings.embed_model = OpenAIEmbedding(model=config.EMBED_MODEL)


def build_index() -> VectorStoreIndex:
    """data/ の文書を読み込んでベクトルインデックスを構築し、
    Chroma に永続化する。構築済みのインデックスを返す。
    """
    # 1) 文書の読み込み。フォルダ内の .md / .txt / .pdf などをまとめて読む
    documents = SimpleDirectoryReader(config.DATA_DIR).load_data()

    # 2) 文書を検索しやすいサイズのチャンクに分割する設定
    #    長すぎると検索がぼやけ、短すぎると文脈が失われるため、
    #    重なり（overlap）を持たせて分割する。
    splitter = SentenceSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
    )

    # 3) 保存先（Chroma）を用意
    vector_store = get_vector_store()
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # 4) 分割 → 埋め込み → 保存までを一括実行
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        transformations=[splitter],
    )
    return index


def main() -> None:
    configure_embeddings()
    print(f"[ingest] '{config.DATA_DIR}' の文書を読み込み、インデックスを構築します...")
    build_index()
    print(f"[ingest] 完了しました。ベクトルDBを '{config.CHROMA_DIR}' に保存しました。")


if __name__ == "__main__":
    main()
