"""ベクトルストア（Chroma）まわりの共通処理。

ingest（書き込み）と rag（読み込み）の両方から同じ設定で
Chroma に接続できるよう、接続処理をここに集約する。
"""
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore

from . import config


def get_vector_store() -> ChromaVectorStore:
    """Chroma の永続クライアントに接続し、LlamaIndex 用の
    ベクトルストアオブジェクトを返す。

    PersistentClient を使うことで、埋め込み結果がディスク
    （config.CHROMA_DIR）に保存され、次回起動時に再利用できる。
    """
    chroma_client = chromadb.PersistentClient(path=config.CHROMA_DIR)
    # コレクションが無ければ作成、あれば取得
    collection = chroma_client.get_or_create_collection(config.COLLECTION_NAME)
    return ChromaVectorStore(chroma_collection=collection)
