#!/bin/bash
# 簡易起動スクリプト

set -e

echo "=== Text-to-Speech Server 起動スクリプト ==="
echo ""

# 仮想環境の確認
if [ ! -d ".venv" ]; then
    echo "⚠️  仮想環境が見つかりません。"
    echo "   make init を実行してください。"
    exit 1
fi

# 仮想環境を有効化
source .venv/bin/activate

# 環境変数の確認
if [ ! -f ".env" ]; then
    echo "⚠️  .envファイルが見つかりません。"
    echo "   APIキーを設定してください。"
    exit 1
fi

# モデルの確認
if [ ! -d "models/deberta-v2-large-japanese-char-wwm" ]; then
    echo "⚠️  BERTモデルが見つかりません。"
    echo "   make init を実行してモデルをダウンロードしてください。"
    exit 1
fi

if [ ! -d "models/jvnv/jvnv-F1-jp" ]; then
    echo "⚠️  TTSモデルが見つかりません。"
    echo "   make init を実行してモデルをダウンロードしてください。"
    exit 1
fi

# サーバー起動
echo "サーバーを起動しています..."
echo "ポート: 8002"
echo "URL: http://localhost:8002/"
echo ""
python -m server.pipecat.text_to_speech
