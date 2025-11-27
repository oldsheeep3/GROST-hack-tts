# Coala Voice Agent

リアルタイム音声対話エージェントのサーバ実装です。FastAPI + WebSocket ゲートウェイで音声フレームを受信し、Deepgram STT → Gemini LLM → Style-Bert-VITS2 TTS までを 1 セッション内でストリーミング処理します。状態遷移は `VADFeatureTracker` と `TurnManager` が一元管理し、バックチャネルや割り込み制御も含めて `PipecatSession` から駆動します。

## プロジェクト概要
- `server/pipecat/main.py` が FastAPI エントリ。WebSocket 1本＝1 `PipecatSession` を生成して音声・イベントをルーティングします。
- `server/agent/services/*` に LLM / STT / TTS / Backchannel サービスをモジュール単位で配置。Style-Bert-VITS2 へのアクセスや Deepgram WebSocket 管理もこの層に集約しています。
- `server/agent/turn_manager.py` と `server/agent/vad_tracker.py` が VAD 主導のターン制御を担当。各種閾値やイベント定義はこのファイルで調整します。
- `docs/architecture.md` と `docs/vad_state_machine.md` にシステム全体とステートマシンの詳細があります。README は日常的なセットアップ手順と外部依存を網羅します。

## ディレクトリのポイント
```
server/
  pipecat/           # FastAPIエントリ + 旧Pipecatラッパ
  agent/
    services/        # STT / LLM / TTS / Backchannel サービス
    turn_manager.py  # 会話ステートマシン
    vad_tracker.py   # Silero VAD + 特徴量
models/
  deberta-v2-large-japanese-char-wwm/  # TTS向けBERT
  jvnv/                                 # Style-Bert-VITS2 音声モデル群
scripts/
  generate_backchannel_audio.py         # 相槌WAV生成ユーティリティ
```

## `.env` と外部APIキー
`server/config.py` が `.env` から各種キーとモデルパスを読み込みます。最低限、以下を設定してください。

```11:35:server/config.py
MODELS_DIR = PROJECT_ROOT / "models"
BERT_MODEL_PATH = MODELS_DIR / "deberta-v2-large-japanese-char-wwm"
TTS_MODEL_DIR = MODELS_DIR / "jvnv/jvnv-F1-jp"
DEEPGRAM_API_KEY = os.environ.get("DEEPGRAM_API_KEY", "")
GEMINI_API_KEY = os.environ.get("GOOGLE_API_KEY", "")
DASHSCOPE_API_KEY = os.environ.get("DASHSCOPE_API_KEY", "")
```

`.env` 例:
```env
GOOGLE_API_KEY=sk-xxxxx            # Gemini 1.5 Flash
DEEPGRAM_API_KEY=dg-xxxxx          # nova-2 streaming
DASHSCOPE_API_KEY=ds-xxxxx         # (任意) Qwen3-ASR 実験用
```

## uv を使った開発環境セットアップ
1. **uv のインストール**
   - macOS: `brew install uv`
   - もしくは公式ワンライナー: `curl -Ls https://astral.sh/uv/install.sh | sh`

2. **仮想環境の作成と有効化**
   ```bash
   cd /path/to/coala
   uv venv --python 3.10 .venv
   source .venv/bin/activate
   ```

3. **Python 依存関係のインストール**
   ```bash
   uv pip install --upgrade pip wheel setuptools
   uv pip install fastapi uvicorn websockets numpy python-dotenv google-genai dashscope
   # Torch (CPU) が必要。GPU を使う場合は適宜 index-url を差し替え。
   uv pip install "torch>=2.2" --index-url https://download.pytorch.org/whl/cpu
   # Pipecat & Style-Bert-VITS2 は Git から取得
   uv pip install "pipecat-ai @ git+https://github.com/pipecat-ai/pipecat.git"
   uv pip install "style-bert-vits2 @ git+https://github.com/litagin/Style-Bert-VITS2.git"
   ```
   - Silero VAD の `torch.hub.load` を使用するため `torch` が必須です。
   - DashScope/Qwen を使わない場合は `dashscope` を省略しても構いません。

4. **uv での実行**
   - Webサーバ起動: `uv run python -m server.pipecat.main` (デフォルト 0.0.0.0:8001)
   - TTS 単体テスト: `uv run python test_tts.py "テキスト"`
   - 相槌生成: `uv run python scripts/generate_backchannel_audio.py`

## Style-Bert-VITS2 の導入とモデル配置
1. **Style-Bert-VITS2 ライブラリ**
   - 上記 `uv pip install "style-bert-vits2 @ git+https://github.com/litagin/Style-Bert-VITS2.git"` でソースを取得します。
   - `style_bert_vits2` モジュールは `server/agent/services/tts_service.py` から使用され、初回呼び出し時に BERT / TTS モデルをロードします。

2. **BERT (日本語 DeBERTa) の配置**
   - Hugging Face から `izumi-lab/deberta-v2-large-japanese-char-wwm` を取得:
     ```bash
     huggingface-cli download izumi-lab/deberta-v2-large-japanese-char-wwm \
       --local-dir models/deberta-v2-large-japanese-char-wwm \
       --local-dir-use-symlinks False
     ```
   - `server/config.py` の `BERT_MODEL_PATH` と同じパスに配置されていることを確認してください。

3. **JVNV 音声モデルの配置**
   - Hugging Face: `tsurumaki/Style-Bert-VITS2-JVNV` からパックを取得。
   - 例: `jvnv-F1-jp` を使用する場合:
     ```bash
     huggingface-cli download tsurumaki/Style-Bert-VITS2-JVNV \
       --local-dir models/jvnv \
       --local-dir-use-symlinks False
     ```
   - `models/jvnv/jvnv-F1-jp/{config.json, jvnv-F1-jp_e160_s14000.safetensors, style_vectors.npy}` が揃っていれば OK。必要に応じて `server/config.py` の `TTS_MODEL_DIR` を他スタイルに切り替えます。

4. **検証**
   ```bash
   uv run python test_tts.py "テスト用の短い文章です"
   ```
   初回は BERT / TTS がウォームアップされるため 20〜30 秒ほど待ってください。実行完了後、`output/test_simple_output.wav` が生成されれば導入成功です。

## Deepgram / Gemini / DashScope の接続
- Deepgram STT: `server/agent/services/stt_service.py` が WebSocket 接続を張ります。`DEEPGRAM_MODEL` や `DEEPGRAM_LANGUAGE` は `server/config.py` で調整可能。
- Gemini LLM: `server/agent/services/llm_service.py` が `google-genai` SDK を利用します。`GEMINI_MODEL`/`SYSTEM_PROMPT` は config 参照。
- DashScope/Qwen: `server/agent/services/asr_service.py` は実験用クライアントです。必要なら `.env` に `DASHSCOPE_API_KEY` を追加してください。

## モデルダウンロードのベストプラクティス
- Hugging Face からのダウンロードは `huggingface-cli login` 後に実行すると高速です。
- モデル更新時は `models/` 以下を上書きコピーするだけで反映できます。`git-lfs` 管理対象ではないため、容量に注意しつつ `.gitignore` のまま保持してください。
- 大規模モデルを CI で扱わない場合、`models/README.md` にローカル配置手順を追記しておくと他メンバーと同期しやすくなります。

## トラブルシューティング
- **Torch import error**: `uv pip install torch --index-url https://download.pytorch.org/whl/cpu` を実行し直す。Apple Silicon で GPU を使う場合は該当 wheel に置き換える。
- **Style-Bert-VITS2 の日本語 BERT 読み込み失敗**: `models/deberta-v2-large-japanese-char-wwm` 配下に `config.json` / `pytorch_model.bin` / `tokenizer_config.json` が揃っているか確認。
- **Deepgram 接続失敗**: `.env` の `DEEPGRAM_API_KEY` を再確認し、`uv run python -m server.pipecat.main` 実行時ログを確認。
- **Gemini レスポンスが遅い**: `server/config.py` の `LLM_MAX_TOKENS` を下げる、または `LLMService.history` を定期的にリセットしてください。

以上の手順を README に集約したので、新規メンバーはこのドキュメントだけで環境構築・モデル準備・サーバ起動まで完結します。
