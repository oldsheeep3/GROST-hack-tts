# Text-to-Speech Server - テキスト入力から音声ストリーム出力

テキスト入力（WebSocket）から音声をバイナリストリームで出力するシンプルなTTSサーバーです。

## 概要

このプロジェクトは、WebSocket経由でテキストメッセージを受け取り、AI（Gemini LLM）が応答を生成し、TTS（Style-Bert-VITS2）で音声化してクライアントにストリーミング配信します。

### システム構成
```
テキスト入力（WebSocket）
    ↓
Gemini LLM（応答生成）
    ↓
文単位セグメント化
    ↓
Style-Bert-VITS2 TTS
    ↓
音声バイナリ出力（PCM int16、ストリーム）
```

### 主要機能
- ✅ WebSocketでテキストメッセージを受信
- ✅ Gemini Flash LLMで応答生成（ストリーミング）
- ✅ 文単位でTTS合成（「。？！」で区切り）
- ✅ 音声をPCM int16バイナリでストリーミング送信
- ✅ ブラウザデモクライアント付属（`/static/text_to_speech.html`）

## クイックスタート

### 1. 環境構築

```bash
# 仮想環境作成
uv venv --python 3.10 .venv
source .venv/bin/activate

# 依存関係インストール（推奨）
uv pip install -r requirements.txt

# または手動インストール
uv pip install --upgrade pip wheel setuptools
uv pip install fastapi uvicorn websockets numpy python-dotenv google-genai
uv pip install "torch>=2.2" --index-url https://download.pytorch.org/whl/cpu
uv pip install style-bert-vits2
```

**注意**: Style-Bert-VITS2は公式pipパッケージを使用（リポジトリ: `https://github.com/litagin02/Style-Bert-VITS2`）

### 2. モデルのダウンロード

#### BERT（日本語）
```bash
hf download ku-nlp/deberta-v2-large-japanese-char-wwm \
  --local-dir models/deberta-v2-large-japanese-char-wwm
```

#### Style-Bert-VITS2 音声モデル（JVNV）
```bash
hf download tsurumaki/Style-Bert-VITS2-JVNV \
  --local-dir models/jvnv
```

### 3. 環境変数設定

`.env` ファイルを作成:
```env
GOOGLE_API_KEY=your-gemini-api-key-here
```

### 4. サーバー起動

**簡単起動（推奨）**:
```bash
# 初期設定（初回のみ）
make init

# サーバー起動
make start
```

**手動起動**:
```bash
# テキスト→音声サーバー（ポート8002）
uv run python -m server.pipecat.text_to_speech

# または元の音声→音声サーバー（ポート8001）
make start-voice
# または
uv run python -m server.pipecat.main
```

### 5. ブラウザでアクセス

```
http://localhost:8002/
```

## Makefileコマンド

| コマンド | 説明 |
|---------|------|
| `make init` | 初期設定（環境構築とモデルダウンロード案内） |
| `make start` | テキスト→音声サーバーを起動（ポート8002） |
| `make start-tts` | テキスト→音声サーバーを起動（明示的） |
| `make start-voice` | 音声→音声サーバーを起動（ポート8001） |
| `make test` | テストを実行 |
| `make clean` | 一時ファイルを削除 |
| `make help` | ヘルプを表示 |

## API仕様

### WebSocketエンドポイント
```
ws://localhost:8002/ws/tts
```

### メッセージフォーマット

#### クライアント → サーバー

**テキスト送信**:
```json
{
  "type": "text",
  "text": "こんにちは！"
}
```

**設定確認**:
```json
{
  "type": "config",
  "mode": "text-to-speech"
}
```

**生成キャンセル**:
```json
{
  "type": "cancel"
}
```

**Ping**:
```json
{
  "type": "ping"
}
```

#### サーバー → クライアント

**イベント（JSON）**:
```json
{
  "type": "event",
  "name": "session_start|text_received|llm_start|tts_start|tts_done|llm_end|error|pong",
  "t": 12345,
  "...": "イベント固有のペイロード"
}
```

**音声データ（バイナリ）**:
- フォーマット: PCM int16（リトルエンディアン）
- サンプリングレート: 16000 Hz
- チャンネル数: 1（モノラル）
- ビット深度: 16bit

### イベント一覧

| イベント名 | 説明 | ペイロード例 |
|-----------|------|-------------|
| `session_start` | セッション開始 | `{"status": "ready"}` |
| `config_ack` | 設定確認応答 | `{"mode": "text-to-speech"}` |
| `text_received` | テキスト受信確認 | `{"text": "こんにちは"}` |
| `llm_start` | LLM生成開始 | `{"user_text": "こんにちは"}` |
| `tts_start` | TTS合成開始 | `{"text": "こんにちは！"}` |
| `tts_done` | TTS合成完了 | `{"text": "...", "duration_ms": 1200, "sample_rate": 16000}` |
| `llm_end` | LLM生成完了 | `{"reason": "completed"}` |
| `error` | エラー発生 | `{"message": "エラー詳細"}` |
| `pong` | Ping応答 | `{}` |

## ディレクトリ構造

```
.
├── server/
│   ├── pipecat/
│   │   ├── main.py                    # 元の音声対話サーバー（ポート8001）
│   │   └── text_to_speech.py          # 新しいテキスト→音声サーバー（ポート8002）
│   ├── agent/
│   │   ├── segmenter.py               # 文単位セグメント化
│   │   └── services/
│   │       ├── llm_service.py         # Gemini LLM
│   │       └── tts_service.py         # Style-Bert-VITS2 TTS
│   ├── static/
│   │   ├── realtime.html              # 音声対話デモ
│   │   └── text_to_speech.html        # テキスト→音声デモ
│   └── config.py                      # 設定ファイル
├── models/
│   ├── deberta-v2-large-japanese-char-wwm/  # BERT
│   └── jvnv/jvnv-F1-jp/               # 音声モデル
├── docs/
│   ├── architecture.md                # 元のアーキテクチャ
│   └── tts_model_recommendations.md   # TTSモデル推奨
└── requirements.txt
```

## 設定

### `server/config.py`

```python
# モデルパス
BERT_MODEL_PATH = MODELS_DIR / "deberta-v2-large-japanese-char-wwm"
TTS_MODEL_DIR = MODELS_DIR / "jvnv/jvnv-F1-jp"

# TTS設定
TTS_DEVICE = "cpu"  # GPU使用時は "cuda"
TTS_DEFAULT_STYLE = "Neutral"  # Happy, Sad なども可能

# Gemini設定
GEMINI_MODEL = "gemini-2.0-flash"
LLM_MAX_TOKENS = 200

# システムプロンプト
SYSTEM_PROMPT = """あなたはかわいいAI VTuber "つるま こあら" です..."""
```

### GPU使用（推奨）

TTS合成を高速化するには GPU を使用してください。

```python
# server/config.py
TTS_DEVICE = "cuda"  # CPUの5〜10倍高速
```

**必要なGPU**:
- NVIDIA GPU（CUDA対応）
- VRAM: 4GB以上推奨

**PyTorch GPU版のインストール**:
```bash
uv pip install "torch>=2.2" --index-url https://download.pytorch.org/whl/cu118
```

## TTS モデルの選択

現在は **Style-Bert-VITS2** を使用していますが、用途に応じて他のモデルも検討できます。

詳細は [`docs/tts_model_recommendations.md`](docs/tts_model_recommendations.md) を参照してください。

### 推奨モデル

| 用途 | 推奨モデル |
|------|-----------|
| 開発・テスト | **Style-Bert-VITS2**（現在使用中） |
| 商用・高品質 | Google Cloud TTS |
| プライバシー重視 | **Style-Bert-VITS2** |
| 低コスト | **Style-Bert-VITS2** or VOICEVOX |

## パフォーマンス

### CPU実行（現在の設定）
- 初期化時間: 約10〜20秒（初回のみ）
- TTS合成速度: 約1〜2秒/文（CPUにより変動）
- メモリ使用量: 約2〜3GB

### GPU実行（推奨）
- 初期化時間: 約5〜10秒
- TTS合成速度: 約0.2〜0.5秒/文（**5〜10倍高速**）
- VRAM使用量: 約2〜3GB

## トラブルシューティング

### モデルのロードに失敗する
```
models/deberta-v2-large-japanese-char-wwm/
models/jvnv/jvnv-F1-jp/
```
が正しくダウンロードされているか確認してください。

### 音声が再生されない
- ブラウザのコンソールでエラーを確認
- WebSocketが正しく接続されているか確認
- サーバーログでTTS合成が完了しているか確認

### レイテンシが高い
- GPUを使用してください（`TTS_DEVICE = "cuda"`）
- LLMの `max_tokens` を減らす（`LLM_MAX_TOKENS = 100`）
- ネットワーク帯域を確認

## ライセンス

- Style-Bert-VITS2: MIT License
- このプロジェクト: MIT License（要確認）

## WebSocket通信

詳細は [`docs/websocket_communication.md`](docs/websocket_communication.md) を参照してください。

### WebSocketエンドポイント
```
ws://localhost:8002/ws/tts
```

### 主要な通信箇所
- **サーバー**: `server/pipecat/text_to_speech.py` 166-213行目
- **クライアント**: `server/static/text_to_speech.html` 188-385行目

### メッセージ形式
- **テキスト**: JSON（イベント、設定、制御）
- **バイナリ**: PCM int16（音声データ、16kHz、モノラル）

## 関連ドキュメント

- [WebSocket通信の仕組み](docs/websocket_communication.md) ⭐ NEW
- [TTS モデル推奨](docs/tts_model_recommendations.md)
- [元のアーキテクチャ](docs/architecture.md)
- [VAD ステートマシン](docs/vad_state_machine.md)
