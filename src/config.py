"""アプリ全体の設定値をまとめたモジュール。

パスやモデル名、検索パラメータなどを一箇所に集約し、
他のモジュール（ingest / rag / app）から参照する。
環境変数があればそちらを優先する（.env を利用）。
"""
import os
from dotenv import load_dotenv

# .env ファイルを読み込む（存在しなくてもエラーにはならない）
load_dotenv()

# ---- パス関連 ----
# このファイル（src/config.py）から見たプロジェクトルート
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# FAQ の元文書を置くフォルダ
DATA_DIR = os.path.join(BASE_DIR, "data")
# Chroma（ベクトルDB）の保存先。ここに埋め込み済みデータが永続化される
CHROMA_DIR = os.path.join(BASE_DIR, "chroma_db")
# Chroma のコレクション名（テーブル名のようなもの）
COLLECTION_NAME = "faq"

# ---- モデル関連 ----
# 回答生成に使う LLM。gpt-4o-mini は安価で日本語も十分実用的
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
# 文書と質問をベクトル化する埋め込みモデル
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")

# ---- 検索・分割パラメータ ----
# 文書を分割する際の1チャンクの文字数の目安（トークン基準）
CHUNK_SIZE = 512
# チャンク間で重複させる量。文脈が途切れるのを防ぐ
CHUNK_OVERLAP = 64
# 質問に対して検索で拾ってくる関連チャンクの数
SIMILARITY_TOP_K = 3

# ---- プロンプト ----
# RAG の肝。「渡した文脈だけを根拠に答える／無ければ正直に分からないと言う」
# よう指示することで、AIが知識をでっち上げる（ハルシネーション）のを抑える。
QA_PROMPT_TEMPLATE = (
    "あなたは人材派遣会社の社内問い合わせに答える、丁寧で正確なアシスタントです。\n"
    "以下の「社内文書」の内容だけを根拠にして、日本語で質問に答えてください。\n"
    "社内文書に書かれていないことは推測せず、"
    "「社内文書には記載が見当たりませんでした。担当のコーディネーターにご確認ください。」"
    "と答えてください。\n"
    "\n"
    "----- 社内文書 -----\n"
    "{context_str}\n"
    "--------------------\n"
    "\n"
    "質問: {query_str}\n"
    "回答:"
)
