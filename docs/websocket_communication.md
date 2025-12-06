# WebSocket通信の仕組み

このプロジェクトでは、クライアント（ブラウザ）とサーバー間でWebSocketを使った双方向通信を行っています。

## 1. テキスト→音声サーバー（text_to_speech.py）

### WebSocketエンドポイント
**ファイル**: `server/pipecat/text_to_speech.py`

```python
@app.websocket("/ws/tts")
async def websocket_text_to_speech(ws: WebSocket):
    """Text-to-Speech WebSocket endpoint."""
    await ws.accept()  # ← WebSocket接続を受け入れ
    
    session = TextToSpeechSession(ws)
    await session.start()
    
    try:
        while True:
            msg = await ws.receive()  # ← クライアントからメッセージを受信
            
            if msg["type"] == "websocket.disconnect":
                break
            
            if msg["type"] == "websocket.receive":
                data = msg.get("text") or msg.get("bytes")
                
                if isinstance(data, str):
                    parsed = json.loads(data)
                    msg_type = parsed.get("type")
                    
                    if msg_type == "text":
                        # テキストメッセージを処理
                        text = parsed.get("text", "")
                        await session.handle_text_message(text)
```

### 送信メソッド（サーバー → クライアント）

#### 1. イベント送信（JSON）
```python
async def send_event(self, name: str, payload: dict = None):
    """イベントをクライアントに送信"""
    msg = {
        "type": "event",
        "name": name,
        "t": round(t),
        **(payload or {})
    }
    await self.ws.send_text(json.dumps(msg))  # ← テキストメッセージとして送信
```

#### 2. 音声送信（バイナリ）
```python
async def send_audio(self, audio: np.ndarray):
    """音声をクライアントに送信（バイナリ）"""
    await self.ws.send_bytes(np_int16_to_bytes(audio))  # ← バイナリメッセージとして送信
```

### 通信フロー

```
クライアント                      サーバー
    |                                |
    |-- WebSocket接続要求 ---------->|
    |<-------- 接続受け入れ ----------|
    |                                |
    |-- {"type":"text", ...} ------->|
    |                                | (LLM処理開始)
    |<- {"name":"llm_start", ...} --|
    |                                | (TTS合成)
    |<- {"name":"tts_start", ...} --|
    |<--- [音声バイナリ] ------------|
    |<- {"name":"tts_done", ...} ---|
    |<- {"name":"llm_end", ...} ----|
    |                                |
```

---

## 2. 音声→音声サーバー（main.py）

### WebSocketエンドポイント
**ファイル**: `server/pipecat/main.py`

```python
@app.websocket("/ws/realtime")
async def websocket_realtime(ws: WebSocket):
    """Real-time voice WebSocket endpoint."""
    await ws.accept()
    
    session = PipecatSession(ws)
    await session.start()
    
    try:
        while True:
            msg = await ws.receive()
            
            if msg["type"] == "websocket.receive":
                data = msg.get("text") or msg.get("bytes")
                
                if isinstance(data, bytes):
                    # 音声フレーム（PCM）を処理
                    await session.handle_audio_frame(data)
```

### 通信フロー

```
クライアント                      サーバー
    |                                |
    |-- WebSocket接続要求 ---------->|
    |<-------- 接続受け入れ ----------|
    |                                |
    |-- [音声PCMバイナリ] ---------->| (マイク音声)
    |                                | (VAD + STT処理)
    |<- {"name":"stt_partial"} -----|
    |                                | (LLM + TTS処理)
    |<- {"name":"tts_start"} --------|
    |<--- [音声バイナリ] ------------|
    |                                |
```

---

## 3. クライアント側（ブラウザ）

### WebSocket接続
**ファイル**: `server/static/text_to_speech.html`

```javascript
// WebSocket接続を確立
const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
const wsUrl = `${protocol}//${window.location.host}/ws/tts`;
ws = new WebSocket(wsUrl);

ws.onopen = () => {
    console.log('WebSocket connected');
    // 設定を送信
    ws.send(JSON.stringify({ type: 'config', mode: 'text-to-speech' }));
};

// メッセージ受信
ws.onmessage = async (event) => {
    if (typeof event.data === 'string') {
        // テキストメッセージ（イベント）
        const msg = JSON.parse(event.data);
        handleEvent(msg);
    } else {
        // バイナリメッセージ（音声）
        const arrayBuffer = await event.data.arrayBuffer();
        await playAudioBuffer(arrayBuffer);
    }
};

// テキスト送信
ws.send(JSON.stringify({ type: 'text', text: 'こんにちは' }));
```

---

## 4. WebSocket通信の種類

### テキストメッセージ（JSON）
- **用途**: イベント通知、設定、制御コマンド
- **形式**: JSON文字列
- **送信**: `ws.send_text(json.dumps(...))`（サーバー）、`ws.send(JSON.stringify(...))`（クライアント）
- **受信**: `json.loads(data)`（サーバー）、`JSON.parse(event.data)`（クライアント）

**例**:
```json
{
  "type": "text",
  "text": "こんにちは"
}
```

### バイナリメッセージ（音声データ）
- **用途**: 音声ストリーミング
- **形式**: PCM int16（16kHz、モノラル）
- **送信**: `ws.send_bytes(audio_bytes)`（サーバー）
- **受信**: `await event.data.arrayBuffer()`（クライアント）

**フォーマット**:
- サンプリングレート: 16000 Hz
- ビット深度: 16bit
- チャンネル: 1（モノラル）
- エンディアン: リトルエンディアン

---

## 5. 主要なWebSocket通信箇所

### サーバー側

| ファイル | 行数 | 説明 |
|---------|------|------|
| `server/pipecat/text_to_speech.py` | 166-213 | テキスト→音声のWebSocketエンドポイント |
| `server/pipecat/text_to_speech.py` | 83 | イベント送信（`ws.send_text`） |
| `server/pipecat/text_to_speech.py` | 90 | 音声送信（`ws.send_bytes`） |
| `server/pipecat/main.py` | 358-403 | 音声→音声のWebSocketエンドポイント |
| `server/agent/services/stt_service.py` | 150-200 | Deepgram WebSocket接続（STT） |

### クライアント側

| ファイル | 行数 | 説明 |
|---------|------|------|
| `server/static/text_to_speech.html` | 188-232 | WebSocket接続とイベント処理 |
| `server/static/text_to_speech.html` | 234-256 | メッセージ受信（テキスト/バイナリ） |
| `server/static/text_to_speech.html` | 368-385 | イベントハンドラ |
| `server/static/realtime.html` | - | 音声→音声のクライアント |

---

## 6. WebSocket接続の確認方法

### ブラウザのDevTools
1. ブラウザでページを開く（http://localhost:8002/）
2. DevToolsを開く（F12）
3. **Network** タブ → **WS**（WebSocket）をクリック
4. `/ws/tts` の接続を確認
5. **Messages** タブで送受信メッセージを確認

### サーバーログ
```bash
# サーバーを起動
make start

# ログで確認
[Session] Initializing LLM...
[Session] Initializing TTS...
[Session] All services initialized
[Text] Received: 'こんにちは！'
[LLM+TTS] Starting with: 'こんにちは！'
[TTS] Synthesizing: 'こんにちは！こあらだよ！'
[TTS] Done: 24000 samples @ 16000Hz, 1.50s
[TTS] Sent audio: 24000 samples
```

---

## 7. WebSocketのメリット

1. **双方向通信**: サーバーからクライアントにプッシュ通知可能
2. **低レイテンシ**: HTTP POLLINGより高速
3. **ストリーミング**: 音声を小分けにして送信可能
4. **リアルタイム**: ユーザー体験が向上

---

## 8. トラブルシューティング

### WebSocket接続エラー
```
WebSocket connection failed: Error during WebSocket handshake
```
→ サーバーが起動しているか確認（`make start`）

### CORS エラー
→ 同じホスト・ポートからアクセスしているか確認

### 音声が再生されない
→ ブラウザコンソールで `AudioContext` のエラーを確認
→ ユーザー操作（ボタンクリック）後に `AudioContext` を初期化
