.PHONY: help init start start-tts start-voice test clean

# デフォルトターゲット
help:
	@echo "利用可能なコマンド:"
	@echo "  make init        - 初期設定（環境構築とモデルダウンロード）"
	@echo "  make start       - テキスト→音声サーバーを起動（ポート8002）"
	@echo "  make start-tts   - テキスト→音声サーバーを起動（ポート8002）"
	@echo "  make start-voice - 音声→音声サーバーを起動（ポート8001）"
	@echo "  make test        - テストを実行"
	@echo "  make clean       - 一時ファイルを削除"

# 初期設定
init:
	@echo "=== 初期設定を開始 ==="
	@echo ""
	@echo "[1/6] huggingface-cliの確認..."
	@bash -c '. .venv/bin/activate 2>/dev/null || true; \
		if ! command -v huggingface-cli >/dev/null 2>&1; then \
			echo "✓ huggingface-cliは仮想環境で利用可能です"; \
		else \
			echo "✓ huggingface-cliは既にインストール済みです"; \
		fi'
	@echo ""
	@echo "[2/6] 仮想環境の作成..."
	uv venv --python 3.10 .venv || true
	@echo ""
	@echo "[3/6] 依存関係のインストール..."
	@bash -c '. .venv/bin/activate && uv pip install --upgrade pip wheel setuptools'
	@bash -c '. .venv/bin/activate && uv pip install "numpy==1.26.4" --force-reinstall'
	@bash -c '. .venv/bin/activate && uv pip install -r requirements.txt' || bash -c '. .venv/bin/activate && uv pip install fastapi uvicorn websockets python-dotenv google-genai'
	@bash -c '. .venv/bin/activate && uv pip install "torch>=2.2" --index-url https://download.pytorch.org/whl/cpu'
	@bash -c '. .venv/bin/activate && uv pip install "pipecat-ai @ git+https://github.com/pipecat-ai/pipecat.git"'
	@bash -c '. .venv/bin/activate && uv pip install style-bert-vits2'
	@bash -c '. .venv/bin/activate && uv pip install "numpy==1.26.4" --force-reinstall'
	@echo ""
	@echo "[4/6] .envファイルの確認..."
	@if [ ! -f .env ]; then \
		echo "GOOGLE_API_KEY=your-gemini-api-key-here" > .env; \
		echo "DEEPGRAM_API_KEY=your-deepgram-api-key-here" >> .env; \
		echo "⚠️  .envファイルを作成しました。APIキーを設定してください。"; \
	else \
		echo "✓ .envファイルが存在します"; \
	fi
	@echo ""
	@echo "[5/6] モデルディレクトリの作成..."
	@mkdir -p models
	@echo ""
	@echo "[6/6] BERTモデルのダウンロード..."
	@if [ ! -d "models/deberta-v2-large-japanese-char-wwm" ] || [ ! -f "models/deberta-v2-large-japanese-char-wwm/config.json" ]; then \
		echo "BERTモデルをダウンロード中..."; \
		bash -c '. .venv/bin/activate && huggingface-cli download ku-nlp/deberta-v2-large-japanese-char-wwm --local-dir models/deberta-v2-large-japanese-char-wwm'; \
		echo "✓ BERTモデルのダウンロードが完了しました"; \
	else \
		echo "✓ BERTモデルは既に存在します"; \
	fi
	@echo ""
	@echo "[7/7] TTSモデルのダウンロード..."
	@if [ ! -d "models/jvnv/jvnv-F1-jp" ] || [ ! -f "models/jvnv/jvnv-F1-jp/config.json" ]; then \
		echo "TTSモデルをダウンロード中（数分かかります）..."; \
		mkdir -p models/jvnv; \
		bash -c '. .venv/bin/activate && python -c "from server.agent.services.tts_service import download_tts_model; download_tts_model()"' && \
		echo "✓ TTSモデルのダウンロードが完了しました" || \
		echo "⚠️  TTSモデルのダウンロードに失敗しました。手動でダウンロードしてください。"; \
	else \
		echo "✓ TTSモデルは既に存在します"; \
	fi
	@echo ""
	@mkdir -p server/static
	@echo "=== ✅ すべての準備が整いました ==="
	@echo ""
	@echo "次のステップ:"
	@echo "1️⃣  .envファイルにAPIキーを設定"
	@echo "   GOOGLE_API_KEY=your-actual-api-key"
	@echo ""
	@echo "2️⃣  サーバーを起動"
	@echo "   make start"

# テキスト→音声サーバーを起動（デフォルト）
start:
	@echo "=== テキスト→音声サーバーを起動 ==="
	@if [ ! -d ".venv" ]; then \
		echo "❌ 仮想環境が見つかりません。"; \
		echo "   make init を実行してください。"; \
		exit 1; \
	fi
	@if [ ! -f ".env" ]; then \
		echo "❌ .envファイルが見つかりません。"; \
		echo "   APIキーを設定してください。"; \
		exit 1; \
	fi
	@if [ ! -d "models/deberta-v2-large-japanese-char-wwm" ]; then \
		echo "❌ BERTモデルが見つかりません。"; \
		echo "   make init を実行してください。"; \
		exit 1; \
	fi
	@if [ ! -d "models/jvnv/jvnv-F1-jp" ]; then \
		echo "⚠️  TTSモデルが見つかりません。"; \
		echo "   初回起動時に自動ダウンロードされます。"; \
	fi
	@echo "ポート: 8002"
	@echo "URL: http://localhost:8002/"
	@echo ""
	@echo "サーバーを起動しています..."
	@bash -c '. .venv/bin/activate && export OMP_NUM_THREADS=1 GLOO_SOCKET_IFNAME=lo && python -m server.pipecat.text_to_speech 2>&1 | grep -v "NNPACK"'

# テキスト→音声サーバーを起動（明示的）
start-tts:
	@echo "=== テキスト→音声サーバーを起動 ==="
	@echo "ポート: 8002"
	@echo "URL: http://localhost:8002/"
	@echo ""
	@bash -c '. .venv/bin/activate && export OMP_NUM_THREADS=1 GLOO_SOCKET_IFNAME=lo && python -m server.pipecat.text_to_speech 2>&1 | grep -v "NNPACK"'

# 音声→音声サーバーを起動
start-voice:
	@echo "=== 音声→音声サーバーを起動 ==="
	@echo "ポート: 8001"
	@echo "URL: http://localhost:8001/"
	@echo ""
	@bash -c '. .venv/bin/activate && export OMP_NUM_THREADS=1 GLOO_SOCKET_IFNAME=lo && python -m server.pipecat.main 2>&1 | grep -v "NNPACK"'

# テストを実行
test:
	@echo "=== テストを実行 ==="
	@bash -c '. .venv/bin/activate && python test_tts.py "テストメッセージです"'

# 一時ファイルを削除
clean:
	@echo "=== 一時ファイルを削除 ==="
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	@echo "✓ クリーンアップが完了しました"
