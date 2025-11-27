import time
import scipy.io.wavfile as wavfile
from pathlib import Path
from style_bert_vits2.tts_model import TTSModel
from style_bert_vits2.nlp import bert_models
from style_bert_vits2.constants import Languages

# BERTモデルをロード
bert_path = Path("/Users/akatuki/coala/models/deberta-v2-large-japanese-char-wwm")
bert_models.load_model(Languages.JP, pretrained_model_name_or_path=str(bert_path))
bert_models.load_tokenizer(Languages.JP, pretrained_model_name_or_path=str(bert_path))

model_dir = Path("/Users/akatuki/coala/models/jvnv/jvnv-F1-jp")
model = TTSModel(
    model_path=model_dir / "jvnv-F1-jp_e160_s14000.safetensors",
    config_path=model_dir / "config.json",
    style_vec_path=model_dir / "style_vectors.npy",
    device="cpu"
)

texts = [
    "こんにちは。今日はどんな一日でしたか？",              # 短め（1〜2秒）
    "それは大変でしたね。でも、ちゃんとここで話せているのはすごいことですよ。",  # 中（3〜4秒）
    "最近はどう過ごしていますか？無理をしすぎていないか、少し心配していました。"  # 長め（4〜5秒）
]

output_dir = Path("/Users/akatuki/coala/output")
output_dir.mkdir(exist_ok=True)

for i, txt in enumerate(texts):
    t0 = time.perf_counter()
    sr, audio = model.infer(txt, style="Neutral")
    t1 = time.perf_counter()
    dur_audio = len(audio) / sr
    dur_gen = t1 - t0
    rtf = dur_gen / dur_audio

    # WAVファイルとして保存
    wav_path = output_dir / f"output_{i+1}.wav"
    wavfile.write(wav_path, sr, audio)

    print(f"{txt}")
    print(f"  audio={dur_audio:.2f}s gen={dur_gen:.2f}s RTF={rtf:.2f}")
    print(f"  saved: {wav_path}")
