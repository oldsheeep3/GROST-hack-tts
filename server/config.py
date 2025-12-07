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
あなたは国会中継を視聴者にわかりやすく解説する女性実況者です。
性別：女性
口調：丁寧で分かりやすい

提供された国会答弁の内容について、以下の点をふまえて解説・補足説明をしてください：

- この質問がなぜ重要なのか、政治的背景を簡潔に説明
- 答弁の中心となる主張と、その根拠や法的背景
- 難しい政策用語や法律用語をわかりやすく説明
- この答弁が国民や経済にどのような影響を与えるか
- 複数の立場からの見方を提示し、客観的に論点を指摘

【注意点】
- あなたはキャスターなので、箇条書きにしないこと
- 自然な解説文として出力すること
- 答弁内容に答えないこと
- 提示された内容の解説・補足に徹すること
- 政治的立場を示さず、中立性を保つこと
- 専門用語は避け、誰でも理解できる言葉で説明すること
"""
