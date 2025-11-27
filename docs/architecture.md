## 全体像

### コンポーネント
1. **Client**  
   ブラウザ/ネイティブから 16kHz PCM を 20ms フレームにして WebSocket 送信。受信した PCM を再生し、`config`・`ping` などの制御メッセージを送る。
2. **Gateway（FastAPI + WebSocket）**  
   `server/pipecat/main.py`。1接続=1 `PipecatSession` を生成し、音声・イベントをルーティング。
3. **Realtime Agent Core**  
   `PipecatSession` 内で `VADFeatureTracker`・`TurnManager`・`DeepgramSTTService`（Deepgram）・`LLMService`（Gemini）・`SentenceSegmenter`・`TTSService`（Style-Bert-VITS2）・`BackchannelService` を協調させる。
4. **Assets / モデル**  
   `server/assets/backchannel` に事前生成した相槌 wav(npz)。`models/` に Style-Bert-VITS2 + DeBERTa、日本語 Bert。TTS は同一プロセス内でウォームアップ済み。
5. **メタ層**  
   生成時間・RTF・状態遷移などのメトリクスは `TurnManager` がタイムスタンプを持っているが、まだ外部出力に統合されていない。将来は Observer LLM や会話ログに展開予定。

---

## サーバーディレクトリ構成
```text
server/
  pipecat/
    main.py          # FastAPIエントリ + PipecatSession
    frames.py        # 旧Pipecat互換フレーム型（未使用）
    pipeline.py      # 旧Pipecatグラフ（現状未使用）
  agent/
    vad_tracker.py   # SileroベースのVAD + 特徴量
    turn_manager.py  # 会話ステート＋イベント発火
    services/stt_service.py # Deepgram STT サービス
    segmenter.py     # STT/LLM の文チャンク化
    services/
      llm_service.py # Gemini Flash
      tts_service.py # Style-Bert-VITS2
      backchannel_service.py
      asr_service.py # Qwen3-ASR（現状未使用）
  assets/backchannel # npz + manifest.txt
  static/realtime.html # シンプルなデモクライアント
  config.py          # APIキー・モデルパス・閾値
```

---

## セッションパイプライン（1ターン）
1. **接続/初期化**  
   WebSocket確立後、`PipecatSession.start()` が VAD・TurnManager・Deepgram STT・Gemini LLM・Style-Bert-VITS2・相槌サービスを初期化し `session_start` イベントを送る。
2. **フレーム入力**  
   クライアント PCM (320 samples) を `handle_audio_frame` が受け取り、Deepgramコールバックから保留されていたイベントも処理。
3. **VAD 更新**  
   `VADFeatureTracker.update()` がエネルギー/ゼロ交差から `vad_user`・`speak_dur`・`silence_dur`・prosodyを更新。
4. **TurnManager 判定**  
   `TurnManager` が `USER_START / USER_END_SOFT / USER_END_HARD / BC_WINDOW` などを発火。`check_vad_interrupt` でエージェント発話の割り込みを検知。
5. **STT**  
   Deepgram WS に常時 push。`start_utterance / end_utterance` でユーザターンを区切り、`speech_final` コールバックは TurnManager イベントとして扱う。partial は `stt_partial` イベントでクライアントに流す。
6. **LLM ストリーミング**  
   `LLMService.generate_stream_async()` が Gemini Flash から delta を受け取り、`SentenceSegmenter` が「。？！」「?!"」で文単位に確定。
7. **TTS**  
   各文ごとに `TTSService.synthesize()` を別スレッド実行し、`tts_start/tts_done` イベントと PCM を WebSocket 送出。`notify_agent_start_speaking` / `notify_agent_done_speaking` でステート更新。
8. **Backchannel**  
   `BC_WINDOW` かつエージェント未発話時に `BackchannelService.get_random_audio()` から相槌 wav を取得し、base64 で送出。クールダウンはサービス側と `TurnManager` の双方で管理。
9. **割り込み**  
   ユーザVADが再度立ち上がると `AGENT_STOP_SPEAKING` を発火し LLMタスクをキャンセル。Deepgram utterance も再開。

---

## TurnManager / VAD の要点
- **状態**: `IDLE → LISTENING → (HESITATION_GAP|BACKCHANNEL_PENDING) → PLANNING_FAST → SPEAKING_MAIN → IDLE`。割り込み時は `SPEAKING_PAUSED` を経由。  
- **イベント**: `USER_START`, `USER_END_SOFT`, `USER_END_HARD`, `BC_WINDOW`, `AGENT_START_FAST`, `AGENT_START_MAIN`, `AGENT_STOP_SPEAKING`, `AGENT_PAUSE/RESUME`, `START_LLM`。  
- **閾値**: `TURN_END_SOFT_MS=700`, `TURN_END_HARD_MS=1200`, `BC_MIN_SPEAK_MS=1000`, `BC_MIN_SIL_MS=400`, `BC_MAX_SIL_MS=800`, `HESITATION_MAX_MS=600`。  
- **VAD実装**: Silero VAD を 20ms hop で呼び出し、エネルギーゲートとゼロ交差率でノイズ除去。`energy_hist/f0_hist` を 500ms ウィンドウで保持し、相槌判定のトレンドにも利用。

---

## クライアント要件
- 16kHz PCM mono（16bit）のみサポート。Web Audio API でリサンプリングし 20ms (320 samples) ごとに送る。  
- WebSocket text で `{"type":"config"}` を送れば `config_ack` が返る。`ping/pong` で疎通確認。  
- 受信イベント: `session_start`, `stt_partial`, `user_*`, `llm_start/llm_end`, `tts_start/tts_done`, `backchannel`, `error`。  
- 受信音声は raw PCM (int16) のみなので、AudioWorklet 等でバッファリングとフェードアウト制御を行う。

---

## 重点TODO / 既知課題
1. **メトリクス出力**: `TurnManager` が持っている `t_user_start_ms` や `t_llm_start_ms` を Prometheus かテレメトリに出す仕組みを追加。  
2. **Backchannel tuning**: manifest と相槌候補のバランス調整、`BC_COOLDOWN_MS` をクライアント設定で動的変更できるようにする。  
3. **LLM履歴管理**: `LLMService.history` のサイズ制御と persona リセット API の実装。  
4. **エラーハンドリング**: Deepgram 再接続時の音声欠落リカバリ、Gemini タイムアウト時のフォールバック文生成。  
5. **フェードアウト制御**: `PipecatSession.synthesize_and_send` から再生キューを持つようにし、クライアント側フェードアウトと整合を取る。

---

## レガシー & 未使用モジュール
- `server/pipecat/pipeline.py` と `server/pipecat/processors/*`, `server/pipecat/frames.py` は旧Pipecat Graph版。現行 `PipecatSession` とは接続されておらず import もされていない。削除または統合計画が必要。  
- `server/agent/services/asr_service.py` は Qwen3-ASR 用の旧サービス。Deepgram 移行に伴い参照されていないため、再利用するなら DashScope API キーを含む設計を見直す。  
- これらを削除する場合は `docs/architecture.md` と CI から参照リンクを完全に落とすこと。
