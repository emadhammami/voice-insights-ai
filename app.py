# Voice Insights AI
# Author: Emad Hammami
#
# A small NLP pipeline that takes an audio recording and gives back three things:
# the full transcript, a condensed summary, and an emotion breakdown.
#
# The motivation was simple — I kept sitting through long meeting recordings
# just to find two key points. This automates that.
#
# Models used:
#   - openai/whisper-small          → speech recognition
#   - sshleifer/distilbart-cnn-12-6 → abstractive summarisation
#   - j-hartmann/emotion-english-distilroberta-base → 7-class emotion scoring
#
# Run:  python app.py   (opens http://127.0.0.1:7860)

import os
import shutil
import tempfile

import torch
import gradio as gr
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

# ------------------------------------------------------------------
# ffmpeg setup
# ------------------------------------------------------------------
# The Whisper pipeline needs ffmpeg to decode audio files.
# We use imageio-ffmpeg which ships its own binary, but the binary
# has a versioned filename like ffmpeg-win-x86_64-v7.1.exe — not
# simply "ffmpeg". We copy it to a temp dir under the expected name
# so any downstream library that calls subprocess("ffmpeg", ...)
# can find it without a system-level install.
import imageio_ffmpeg as _imageio_ffmpeg

_src = _imageio_ffmpeg.get_ffmpeg_exe()
_bin_dir = tempfile.mkdtemp(prefix="vi_ffmpeg_")
_dst = os.path.join(_bin_dir, "ffmpeg.exe" if os.name == "nt" else "ffmpeg")
shutil.copy2(_src, _dst)
os.chmod(_dst, 0o755)
os.environ["PATH"] = _bin_dir + os.pathsep + os.environ.get("PATH", "")

# ------------------------------------------------------------------
# Device
# ------------------------------------------------------------------
device = 0 if torch.cuda.is_available() else -1
print(f"[startup] device: {'GPU' if device == 0 else 'CPU'}")

# ------------------------------------------------------------------
# Model loading  (done once at import time, not per request)
# ------------------------------------------------------------------
print("[startup] loading whisper-small ...")
asr = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-small",
    device=device,
)

# transformers ≥ 5.x removed 'summarization' from the pipeline registry,
# so we load the seq2seq model directly.
print("[startup] loading distilbart-cnn-12-6 ...")
_tok = AutoTokenizer.from_pretrained("sshleifer/distilbart-cnn-12-6")
_bart = AutoModelForSeq2SeqLM.from_pretrained("sshleifer/distilbart-cnn-12-6")
if device == 0:
    _bart = _bart.cuda()

print("[startup] loading emotion classifier ...")
emotion_clf = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    top_k=None,
    device=device,
)

print("[startup] all models ready.\n")

# Emotion → emoji, purely cosmetic
_EMOJI = {
    "joy": "😄", "sadness": "😢", "anger": "😠",
    "fear": "😨", "surprise": "😲", "disgust": "🤢", "neutral": "😐",
}


# ------------------------------------------------------------------
# Core functions
# ------------------------------------------------------------------

def split_into_chunks(text: str, max_words: int = 180) -> list[str]:
    """
    Split text into word-bounded chunks of at most max_words words.

    DistilBART has a 1024-token hard limit. 180 words is a comfortable
    budget that leaves room for tokeniser overhead.
    """
    words = text.split()
    return [" ".join(words[i:i + max_words]) for i in range(0, len(words), max_words)]


def transcribe(audio_path: str) -> str:
    """
    Convert an audio file to text with Whisper.

    chunk_length_s=30 enables long-form mode — Whisper slides a 30-second
    window across the file internally, so there's no length restriction on
    the input we pass.
    """
    out = asr(audio_path, chunk_length_s=30, batch_size=8)
    return out["text"].strip()


def summarise(text: str) -> str:
    """
    Produce a concise summary of the transcript text.

    For transcripts longer than 180 words we summarise each chunk separately
    and join the results. This keeps each batch within the model's token budget
    while still covering the full recording.
    """
    if len(text.split()) < 30:
        return text  # too short to be worth summarising

    results = []
    for chunk in split_into_chunks(text):
        if len(chunk.split()) < 10:
            continue  # discard tiny leftover chunks
        enc = _tok(chunk, return_tensors="pt", max_length=1024, truncation=True)
        if device == 0:
            enc = {k: v.cuda() for k, v in enc.items()}
        ids = _bart.generate(
            enc["input_ids"],
            max_length=130,
            min_length=30,
            num_beams=4,
            do_sample=False,
        )
        results.append(_tok.decode(ids[0], skip_special_tokens=True))

    return " ".join(results)


def emotion_report(text: str) -> str:
    """
    Run the emotion classifier and format results as an ASCII bar chart.

    The model has a 512-token ceiling, so we feed it the first ~400 words.
    That's usually enough to capture the dominant emotional tone of a piece.
    """
    sample = " ".join(text.split()[:400])
    scores = sorted(emotion_clf(sample)[0], key=lambda x: x["score"], reverse=True)

    lines = []
    for s in scores:
        label = s["label"]
        pct = s["score"] * 100
        bar = "█" * int(pct / 5)
        lines.append(f"{_EMOJI.get(label, '')} {label:<10} {bar:<20} {pct:5.1f}%")

    return "\n".join(lines)


def run_pipeline(audio_file):
    """Entry point called by the Gradio button."""
    if audio_file is None:
        return "Please upload an audio file.", "", ""

    try:
        text = transcribe(audio_file)
        if not text:
            return "Transcription came back empty — is there speech in the file?", "", ""
        return text, summarise(text), emotion_report(text)
    except Exception as err:
        return f"Error: {err}", "", ""


# ------------------------------------------------------------------
# UI
# ------------------------------------------------------------------
with gr.Blocks(title="Voice Insights AI") as demo:

    gr.Markdown("""
    ## 🎙️ Voice Insights AI
    Upload a recording and get a transcript, a summary, and an emotion breakdown — no API keys, runs locally.
    """)

    with gr.Row():
        with gr.Column(scale=1):
            audio_input = gr.Audio(type="filepath", label="Audio file (MP3 or WAV)")
            run_btn = gr.Button("Analyse", variant="primary")
            gr.Markdown("_CPU inference can take 20–60 s depending on file length._")

        with gr.Column(scale=2):
            out_transcript = gr.Textbox(label="Transcript", lines=10, placeholder="transcript appears here")

    with gr.Row():
        out_summary = gr.Textbox(label="Summary", lines=5, placeholder="summary appears here")
        out_emotions = gr.Textbox(label="Emotions", lines=9, placeholder="emotion scores appear here")

    run_btn.click(
        fn=run_pipeline,
        inputs=[audio_input],
        outputs=[out_transcript, out_summary, out_emotions],
    )


if __name__ == "__main__":
    # share=True would generate a public Gradio tunnel URL valid for 72 h.
    # Useful for a quick remote demo, but False is fine for local work.
    demo.launch(share=False, theme=gr.themes.Soft())
