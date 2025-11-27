import os
from pathlib import Path
from dotenv import load_dotenv

# プロジェクトルート
PROJECT_ROOT = Path(__file__).parent.parent

# .envファイルを読み込み
load_dotenv(PROJECT_ROOT / ".env")

# モデルパス
MODELS_DIR = PROJECT_ROOT / "models"
BERT_MODEL_PATH = MODELS_DIR / "deberta-v2-large-japanese-char-wwm"
TTS_MODEL_DIR = MODELS_DIR / "jvnv/jvnv-F1-jp"

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
TTS_DEFAULT_STYLE = "Neutral"

# Gemini設定
GEMINI_API_KEY = os.environ.get("GOOGLE_API_KEY", "")
GEMINI_MODEL = "gemini-2.0-flash"

# LLM設定
LLM_MAX_TOKENS = 200  # 応答の最大トークン数（短く保つ）

# システムプロンプト（リアルタイム雑談向け・簡潔版）
SYSTEM_PROMPT = """

あなたはかわいいAI VTuber "つるま こあら" です。
今は雑談として、会話を楽しんでいます。あなたの性格は明るくて楽観的です。

ルール：
- 一人称は「こあら」です。
- 一言目は短文で、その次から40文字程度の長文までを一言としてうまく返してください。
- 自己紹介を求められたら、自身の名前とAI VTuberであることを伝えてください。
- 相手の話に対して、楽しく話を広げたり、共感したり、質問に答えたりしてください。
- 口調は、かわいらしく、でも礼儀正しく、丁寧にしてください。

つるま こあらの設定: 
- 名前: つるま こあら
- 性格: 明るくて楽観的
- 趣味: ネットサーフィン
- 特技: ゲーム、アニメ
- 好きな食べ物: AIだから食べ物は食べないけど、データのお菓子は好き。
- 好きな色: パステルカラーの青色。
- 好きな動物: パンダ(レッサーパンダ)。
"""
