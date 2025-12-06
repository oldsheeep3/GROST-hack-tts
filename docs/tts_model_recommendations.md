# TTSモデルの検討と推奨

## 現在のモデル: Style-Bert-VITS2

### 概要
- **モデル**: Style-Bert-VITS2（JVNV音声モデル）
- **BERT**: deberta-v2-large-japanese-char-wwm
- **音声モデル**: jvnv-F1-jp
- **デバイス**: CPU（`TTS_DEVICE = "cpu"`）

### メリット
1. **高品質な日本語音声**: 自然な韻律とイントネーション
2. **スタイル制御**: Neutral, Happy, Sad などのスタイル切り替えが可能
3. **オープンソース**: HuggingFaceで公開されており無料で利用可能
4. **ローカル実行**: APIキー不要、プライバシー保護

### デメリット
1. **初期化時間**: BERTとTTSモデルのロードに時間がかかる（初回のみ）
2. **メモリ消費**: モデルが大きい（BERT約1GB + TTS約500MB）
3. **CPU推論の遅さ**: GPUがないとリアルタイム性が低下
4. **ストリーミング非対応**: 文単位で完全に合成してから出力

## 代替モデルの検討

### 1. Google Cloud Text-to-Speech API
**料金**: $4.00 / 100万文字（Standard）、$16.00 / 100万文字（WaveNet/Neural2）

**メリット**:
- 超低レイテンシ（APIコール→音声生成が数百ms）
- ストリーミング対応
- 多様な日本語音声（男性・女性・年齢・方言）
- インフラ管理不要

**デメリット**:
- 従量課金（コストが発生）
- インターネット接続必須
- プライバシー懸念（テキストをGoogleに送信）

**推奨度**: ⭐⭐⭐⭐☆（商用・高品質重視の場合）

### 2. Azure Speech Service (Neural TTS)
**料金**: $16.00 / 100万文字（Neural）

**メリット**:
- Google同様の高品質・低レイテンシ
- SSML対応（感情・速度・ピッチ制御）
- ストリーミング対応
- 日本語音声の選択肢が豊富

**デメリット**:
- 従量課金
- Microsoft依存

**推奨度**: ⭐⭐⭐⭐☆（エンタープライズ向け）

### 3. OpenAI TTS API
**料金**: $15.00 / 100万文字（tts-1）、$30.00 / 100万文字（tts-1-hd）

**メリット**:
- 超高品質な音声（特にtts-1-hd）
- 6種類の音声（alloy, echo, fable, onyx, nova, shimmer）
- 実装が簡単

**デメリット**:
- **日本語対応が微妙**（英語ベースなので日本語の発音が不自然）
- ストリーミング非対応（2024年12月時点）
- 高コスト

**推奨度**: ⭐⭐☆☆☆（日本語用途には非推奨）

### 4. ElevenLabs TTS API
**料金**: $5.00 / 月（10,000文字）〜 $99.00 / 月（500,000文字）

**メリット**:
- 最高品質の音声クローン技術
- 感情表現が豊富
- ストリーミング対応

**デメリット**:
- **日本語対応が限定的**（主に英語特化）
- 高コスト
- カスタム音声の学習が必要

**推奨度**: ⭐⭐☆☆☆（日本語用途には非推奨）

### 5. Coqui TTS（ローカル・オープンソース）
**料金**: 無料（オープンソース）

**メリット**:
- 完全無料
- ローカル実行
- カスタムモデルのトレーニング可能
- 多言語対応（日本語含む）

**デメリット**:
- **日本語の品質がStyle-Bert-VITS2より低い**
- モデルの選択肢が少ない
- メンテナンスが終了（2023年でプロジェクト停止）

**推奨度**: ⭐⭐☆☆☆（現在は非推奨）

### 6. VOICEVOX / SHAREVOX（ローカル・オープンソース）
**料金**: 無料

**メリット**:
- 日本語特化で高品質
- 完全無料・ローカル実行
- キャラクター音声が豊富
- WebSocket API対応

**デメリット**:
- 別プロセスでVOICEVOXエンジンを起動する必要がある
- スタイル制御が限定的
- リアルタイム性が低い（音声合成に時間がかかる）

**推奨度**: ⭐⭐⭐☆☆（無料・日本語重視の場合）

### 7. Bark（ローカル・オープンソース）
**料金**: 無料

**メリット**:
- 多言語対応（日本語含む）
- 感情表現・効果音も生成可能
- 完全無料

**デメリット**:
- **生成速度が遅い**（GPUでも数秒かかる）
- 日本語の品質が不安定
- メモリ消費が大きい

**推奨度**: ⭐⭐☆☆☆（実験用途のみ）

## 推奨モデルの選択基準

| 用途 | 推奨モデル | 理由 |
|------|-----------|------|
| **開発・テスト** | Style-Bert-VITS2 | 無料、高品質、ローカル実行 |
| **商用・高品質** | Google Cloud TTS | 低レイテンシ、ストリーミング対応 |
| **プライバシー重視** | Style-Bert-VITS2 | ローカル実行、データ送信なし |
| **低コスト** | Style-Bert-VITS2 or VOICEVOX | 完全無料 |
| **リアルタイム性** | Google Cloud TTS or Azure Speech | APIの高速性 |

## 最終推奨

### このプロジェクトには **Style-Bert-VITS2** を継続使用することを推奨

**理由**:
1. **無料**: APIコスト0円
2. **プライバシー**: ユーザーデータを外部送信しない
3. **日本語品質**: VTuber・AI音声として十分な自然さ
4. **カスタマイズ性**: スタイル切り替え可能

**改善案**:
- GPUサーバーで実行すれば推論速度が大幅改善（CPU比で5〜10倍高速）
- モデルをRAMに常駐させることで初期化時間を削減（既に実装済み）
- 文単位のストリーミングで体感レイテンシを改善（既に実装済み）

**GPUでの実行例**:
```python
# server/config.py
TTS_DEVICE = "cuda"  # CPUからGPUに変更
```

**必要なGPU**:
- NVIDIA GPU（CUDA対応）
- VRAM: 4GB以上推奨（8GB以上で快適）

**参考**:
- AWS EC2 g4dn.xlarge: $0.526/時間（T4 GPU、16GB VRAM）
- Google Cloud n1-standard-4 + T4: $0.35〜0.50/時間
- ローカル: RTX 3060（12GB VRAM）で十分

---

## 実装の切り替え方法

Style-Bert-VITS2から他のTTSに切り替える場合は、`server/agent/services/tts_service.py` を修正します。

### Google Cloud TTS への切り替え例:
```python
from google.cloud import texttospeech
import numpy as np

class TTSService:
    def __init__(self):
        self.client = texttospeech.TextToSpeechClient()
        self.voice = texttospeech.VoiceSelectionParams(
            language_code="ja-JP",
            name="ja-JP-Neural2-B"  # 女性音声
        )
        self.audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.LINEAR16,
            sample_rate_hertz=16000
        )
    
    def synthesize(self, text: str) -> tuple[int, np.ndarray]:
        synthesis_input = texttospeech.SynthesisInput(text=text)
        response = self.client.synthesize_speech(
            input=synthesis_input,
            voice=self.voice,
            audio_config=self.audio_config
        )
        audio = np.frombuffer(response.audio_content, dtype=np.int16)
        return 16000, audio
```

**環境変数**:
```bash
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account-key.json"
```

**依存関係**:
```bash
uv pip install google-cloud-texttospeech
```
