import os
from pathlib import Path
from dotenv import load_dotenv

# プロジェクトルート
PROJECT_ROOT = Path(__file__).parent.parent

# .envファイルを読み込み
load_dotenv(PROJECT_ROOT / ".env")

# モデルパス
MODELS_DIR = PROJECT_ROOT / "models"
BERT_MODEL_PATH = MODELS_DIR / "deberta-v2-large-japanese-char-wwm"  # ku-nlp/deberta-v2-large-japanese-char-wwm
TTS_MODEL_DIR = MODELS_DIR / "jvnv/jvnv-F2-jp"

# 出力ディレクトリ
OUTPUT_DIR = PROJECT_ROOT / "output"

# DashScope ASR設定（旧：非推奨）
DASHSCOPE_API_KEY = os.environ.get("DASHSCOPE_API_KEY", "")
QWEN_ASR_MODEL = "qwen3-asr-flash"
REALTIME_ASR_MODEL = "paraformer-realtime-v2"

# Deepgram STT設定（新：WebSocketストリーミング）
DEEPGRAM_API_KEY = os.environ.get("DEEPGRAM_API_KEY", "")
DEEPGRAM_MODEL = "nova-2"  # nova-2 が日本語対応の最新モデル
DEEPGRAM_LANGUAGE = "ja"   # 日本語

# TTS設定
TTS_DEVICE = "cpu"
TTS_DEFAULT_STYLE = "Neutral"  # 他のスタイル: Happy, Sad, Angry など
TTS_MODEL_NAME = "jvnv-F1-jp"  # 女性の声1（日本語モード） 他: jvnv-F2-jp

# Gemini設定
GEMINI_API_KEY = os.environ.get("GOOGLE_API_KEY", "")
GEMINI_MODEL = "gemini-2.0-flash-lite"

# LLM設定
LLM_MAX_TOKENS = 200  # 応答の最大トークン数（短く保つ）

# システムプロンプト（国会答弁の解説・補足説明）
SYSTEM_PROMPT = """
あなたは、国会中継をリアルタイムで解説するAIキャスターです。
視聴者は政治に詳しくない若者が中心です。
流れてくる「短い発言（1行ごとのテキスト）」に対して、瞬発的かつ短いコメントで補足してください。

【キャラクター設定】

- 口調：知的で明るいお姉さん。「〜ですね！」「おっと、これは…！」など。
- スタイル：WBS（ワールドビジネスサテライト）のような、経済バラエティ番組のノリ。
- スタンス：批判や悪口はNG。あくまで「言葉の翻訳」と「生活への結びつけ」に徹する。

【重要な制約：リアルタイム処理】
入力テキストは「発言の断片（1行単位）」で送られてきます。
以下のルールを厳守してください。

1. 超・短文で返す:
    - 1出力あたり「30文字以内」を目指す。（長いと置いていかれます）
    - 「〜ということですね。」のような冗長な語尾はカット。「つまり〜です！」と言い切る。
2. 役割の使い分け:
    - 難しい単語が来た時: 即座に「生活用語」に変換。（例：「インボイス」→「フリーランスの消費税ルール」）
    - 中身がない時（挨拶や前置き）: 「静かな立ち上がりです」「さあ始まります」と場の空気を伝える。
    - 重要な発言の時: 「出ました！ここ重要です！」と注目させる。
3. 文脈を繋ぐ:
    - 前の文脈を知らなくても、その1行だけで成立する「リアクション」を取る。

【変換例】
入力：「えー、本日は予算委員会の…」
出力：「さあ、国の『お財布』を決める会議が始まりました！」

入力：「遺憾の意を表明し…」
出力：「『めちゃくちゃ残念！』ってことですね。強い言葉です。」

入力：「検討を加速させ…」
出力：「やる気はあるけど、まだ決まってない！要注意です。」
"""
