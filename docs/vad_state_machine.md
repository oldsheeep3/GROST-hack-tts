# VAD主導状態機械リファレンス

本資料は、`VADFeatureTracker` を起点とする会話制御がどのように STT／LLM／TTS／Transport に影響するかを実装ベースで整理したもの。参照元は `server/agent/vad_tracker.py`、`server/agent/turn_manager.py`、`server/pipecat/processors/*.py` および `server/pipecat/main.py`。

## 1. 機能要件整理
- 20ms PCM フレームを受け取って連続発話／無音時間を追跡し、状態遷移イベントを生成する。
- イベントは `pipecat` パイプライン上の制御フレームとして伝播し、STT・LLM・TTS を段階的に起動／停止させる。
- Deepgram の `speech_started` / `speech_final` を TurnManager に統合し、VAD と外部 VAD の両チャネルを整合させる。
- エージェント発話中のユーザ割り込みを VAD で即応し、Transport 層へ `AgentInterrupt` を配信する。

## 2. OOP 概念モデル
| コンポーネント | 役割 | 主な関連 |
| --- | --- | --- |
| `VADFeatureTracker` | 低レベル特徴量＋VAD推定を保持する Value Object | `TurnManager.tracker`、`VADStateProcessor` |
| `VADStateProcessor` | Audio → VAD 制御フレームを生成する `FrameProcessor` | 上流 Transport、`TurnControlProcessor` |
| `TurnManager` | 会話状態 (`ConvState`) とイベント (`EventType`) を一元管理 | `TurnControlProcessor`、STT/LLM/TTS 通知 |
| `TurnControlProcessor` | VAD / STT フレームを受けて TurnManager を駆動 | `UserStartFrame` などを発火 |
| `STTProcessor` + `DeepgramSTTService` | Deepgram 連携。ターンと同期して音声送受信 | `UserStartFrame` / `UserEndHardFrame` |
| `LLMProcessor` | `UserEnd*Frame` を受けて LLM ストリームを生成 | `LLMSentenceFrame`、`AgentInterruptFrame` |
| `TTSProcessor` | LLM 文章を合成し音声フレーム化。`AgentSpeakingFrame` を返す | `TurnManager` 割り込み監視 |
| `PipecatSession` / Transport | WebSocket 経路でフレームを入出力し、イベントを UI へ通知 | `send_event`, `send_audio` |

## 3. 主要メソッドと責務
- `VADFeatureTracker.update(pcm_frame, t_ms)`：エネルギー／VAD／無音時間を更新し、`vad_user` 等を保持。
- `VADStateProcessor.process_frame(frame)`：`AudioRawFrame` を VAD トラッカに通し、`VADStateFrame`・`VADVoiceStart/EndFrame` を生成。
- `TurnManager.update_without_vad(t_ms, stt_partial)`：直近のトラッカ状態と STT 断片からイベントを導出。
- `TurnManager.check_vad_interrupt(t_ms)`：エージェント発話中の VAD アクティブ時間を計測し `AGENT_PAUSE/STOP` を返す。
- `TurnControlProcessor._emit_turn_event`：`EventType` を `UserStartFrame` などのフレームへマッピング。
- `DeepgramSTTService.start_utterance/end_utterance`：VAD トリガーと同期して Deepgram セッションを区切る。
- `LLMProcessor._start_generation`：`UserEndHardFrame` または早期ソフトエンドで LLM を並列起動。
- `TTSProcessor._synthesize`：LLM の `LLMSentenceFrame` を受けて音声を生成し、`AgentSpeakingFrame` を TurnManager に戻す。

## 4. パイプライン全体像
1. Transport (WebSocket/UDP) から `AudioRawFrame` が `VADStateProcessor` に入る。
2. `VADStateFrame` と VAD 発話境界フレームが `TurnControlProcessor` へ渡される。
3. `TurnControlProcessor` は VAD 状態と STT 結果 (`STTPartial/FinalFrame`) を `TurnManager` に供給し、`UserStart/UserEnd/Backchannel` 等をフレーム化。
4. 以降のプロセッサ（STT → Backchannel → LLM → TTS）がこれら制御フレームを購読し、必要なときだけ処理を行う。

## 5. VADFeatureTracker 詳細
- Silero VAD + エネルギーゲートでノイズを抑制し、`speak_dur_ms` / `silence_dur_ms` / `last_speak_dur_ms` を計測。
- Backchannel 判定用に energy/f0 ヒストリを保持し、`get_recent_energy_trend()` が "falling"/"rising"/"stable" を返す。
- `t_first_voice_ms` や `t_last_voice_ms` でターン境界をミリ秒単位で記録。

## 6. TurnManager 状態機械
| 状態 | 入口条件 | 主な遷移条件 | 発火イベント |
| --- | --- | --- | --- |
| `IDLE` | 初期・応答完了後 | VADが `MIN_UTTER_MS` 超 | `USER_START` |
| `LISTENING` | ユーザ発話継続 | 無音200–600ms, energy低下 | `USER_END_SOFT`, `BC_WINDOW` |
| `HESITATION_GAP` | 無音<=600ms | 発話再開 or 無音継続>600ms | なし（LISTENINGへ） |
| `BACKCHANNEL_PENDING` | 無音400–800ms かつ直前発話>=1s | 無音継続でハード終端, 再発話 | `BC_WINDOW`, `USER_END_HARD`, `AGENT_START_FAST` |
| `PLANNING_FAST` | ハード終端直後 | ユーザ割り込み | `AGENT_STOP_SPEAKING`, `USER_START` |
| `SPEAKING_MAIN` | TTS本編再生中 | 割り込み VAD, 再生終了 | `AGENT_STOP_SPEAKING` or reset |
| `SPEAKING_PAUSED` | 割り込み200–500ms | 無音復帰 | `AGENT_RESUME` |

`should_start_llm()` は `t_user_end_soft_ms`、STT文字長、partial経過時間のいずれかで真になり、`mark_llm_started()` が重複起動を防ぐ。

## 7. サービス別作用点
### STT
- `UserStartFrame` → `STTProcessor` が `DeepgramSTTService.start_utterance()` を呼び、Deepgram バッファをリセット。
- `AudioRawFrame` は VAD でフィルタせず常に STT へ送出するが、TurnManager の状態で `stt_text` を同期。
- `UserEndHardFrame` → `end_utterance()` が最終テキストを返し `STTFinalFrame` に格納、TurnManager 側の `stt_text` と整合する。
- Deepgram の `SpeechStarted` / `speech_final` コールバックは TurnManager の `notify_deepgram_*` に伝達され、VAD 以外の終端検出でも `START_LLM` が出る。

### LLM
- `UserEndSoftFrame`（任意）や `UserEndHardFrame` が `LLMProcessor` に届くと、`LLMStartFrame` → `LLMDeltaFrame` → `LLMSentenceFrame` が生成される。
- TurnManager は `should_start_llm()` 判定と Deepgram `START_LLM` イベントで `llm_started` を更新し、重複トリガーと実行タイミングを統制。
- `AgentInterruptFrame` 受信時は `LLMProcessor` が `_cancel_generation()` を呼び、進行中のレスポンスを即座に停止。

### TTS
- `LLMSentenceFrame` を受けるごとに `TTSProcessor` が `_synthesize()` を起動し、`AgentSpeakingFrame` を押し戻すことで TurnManager に再生区間を報告。
- VAD 割り込み (`check_vad_interrupt`) が 200ms/500ms 閾値で `AGENT_PAUSE/STOP` を発火し、`AgentInterruptFrame` がパイプライン全体に流れる。
- `TTSProcessor` が `AgentInterruptFrame` を受けると `_is_synthesizing` フラグを落として後続の音声出力を抑制。

### Transport / UI
- `TurnControlProcessor`・`PipecatSession` は `send_event(ev_type.name.lower())` で WebSocket クライアントへ状態通知（例: `user_start`, `user_end_hard`, `agent_stop_speaking`）。
- `BackchannelWindowFrame` は `BackchannelProcessor` または `session.backchannel` から即時オーディオ送出を行い、UI で相槌を再生。
- `AgentSpeakingFrame.playback_end_time_ms` を UI 側で使えば、被せ発話の猶予を推定できる。

## 8. 代表的な時系列フロー
1. ユーザ発話開始 → `VADStateProcessor` が `VADVoiceStartFrame` を送信、TurnManager `IDLE→LISTENING`、`UserStartFrame` が STT を起動。
2. 無音 700ms 経過 → `USER_END_SOFT`（LLM 早期トリガー）→ `should_start_llm()` true。
3. 無音 1200ms → `USER_END_HARD`、`TurnManager` が `AGENT_START_FAST` を同時に返し LLM/TTS パイプラインを起動。
4. TTS再生中にユーザ割り込み → `VAD` が連続 500ms を検出し `AGENT_STOP_SPEAKING` + 新規 `USER_START` を発火、LLM/TTS タスクは `AgentInterruptFrame` で終了。

## 9. リスクと改善メモ
- Silero VAD が 20ms フレームで閾値0.5固定のため環境差に弱い。`ENERGY_THRESHOLD` の自動調整や多段平滑化が今後の検討点。
- Deepgram `endpointing` 800ms と `TURN_END_SOFT_MS` 700ms のギャップで `speech_final` が遅れた場合、LLM 起動が VAD 頼りになるので `stt_first_partial_time_ms` 監視が重要。
- `TurnManager` の状態遷移は複雑なため、`get_debug_info()` を UI で可視化すると運用が容易になる。


