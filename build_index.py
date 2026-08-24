"""インデックス構築用のコマンドラインエントリポイント。

実行:
    python build_index.py

data/ の文書を読み込み、埋め込みベクトルに変換して
Chroma（chroma_db/）に保存する。初回起動前に一度実行する。
"""
from src.ingest import main

if __name__ == "__main__":
    main()
