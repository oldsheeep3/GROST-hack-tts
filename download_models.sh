#!/bin/bash
# モデルダウンロードスクリプト

set -e

echo "=== モデルダウンロードスクリプト ==="
echo ""

# ディレクトリ作成
mkdir -p models

# BERT モデル
if [ ! -d "models/deberta-v2-large-japanese-char-wwm" ]; then
    echo "[1/2] BERTモデルをダウンロード中..."
    echo "リポジトリ: ku-nlp/deberta-v2-large-japanese-char-wwm"
    hf download ku-nlp/deberta-v2-large-japanese-char-wwm \
        --local-dir models/deberta-v2-large-japanese-char-wwm
    echo "✓ BERTモデルのダウンロードが完了しました"
else
    echo "[1/2] BERTモデルは既に存在します"
fi

echo ""

# TTS モデル
if [ ! -d "models/jvnv/jvnv-F1-jp" ]; then
    echo "[2/2] TTSモデルをダウンロード中..."
    echo "リポジトリ: tsurumaki/Style-Bert-VITS2-JVNV"
    hf download tsurumaki/Style-Bert-VITS2-JVNV \
        --local-dir models/jvnv
    echo "✓ TTSモデルのダウンロードが完了しました"
else
    echo "[2/2] TTSモデルは既に存在します"
fi

echo ""
echo "=== すべてのモデルのダウンロードが完了しました ==="
