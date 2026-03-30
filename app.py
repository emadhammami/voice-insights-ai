"""
Voice Insights AI
=================
Drop in an audio file and walk away with a full transcript, a tight summary,
and a breakdown of the emotions behind the words — all in one click.

How to run:
  python app.py          → opens the app at http://127.0.0.1:7860

How to deploy:
  Push this folder to a Hugging Face Space (SDK: Gradio) and you're live.
"""

import os
import torch
import gradio as gr
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

# ─── FFmpeg Bootstrap ─────────────────────────────────────────────────────────
# Whisper (via librosa/soundfile) needs ffmpeg to decode MP3/WAV files.
# imageio-ffmpeg ships its own binary so we never need a system-level install.
import imageio_ffmpeg
_ffmpeg_dir = os.path.dirname(imageio_ffmpeg.get_ffmpeg_exe())
os.environ["PATH"] = _ffmpeg_dir + os.pathsep + os.environ.get("PATH", "")

# ─── Device Setup ─────────────────────────────────────────────────────────────
# Use the GPU if one is around — it makes a noticeable difference on long files.
# No GPU? No problem. Everything runs on CPU just fine, just a bit slower.
device = 0 if torch.cuda.is_available() else -1
device_label = "GPU (CUDA)" if device == 0 else "CPU"
print(f"Running on: {device_label}")

# ─── Load Models Once at Startup ──────────────────────────────────────────────
# We load everything here — before the first request comes in — so the UI
# feels snappy once it opens. The first run will download the model weights
# from Hugging Face (~1 GB total); after that they're cached locally.
print("Loading transcription model (Whisper small)…")
transcriber = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-small",
    device=device,
)

print("Loading summarization model (DistilBART)…")
# The pipeline task registry no longer includes 'summarization' or
# 'text2text-generation' in recent transformers builds, so we load
# the tokenizer and model directly — same result, no registry needed.
_sum_tokenizer = AutoTokenizer.from_pretrained("sshleifer/distilbart-cnn-12-6")
_sum_model = AutoModelForSeq2SeqLM.from_pretrained("sshleifer/distilbart-cnn-12-6")
if device == 0:
    _sum_model = _sum_model.cuda()

print("Loading emotion-detection model…")
emotion_detector = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    top_k=None,   # return scores for ALL emotion labels
    device=device,
)

print("All models loaded — ready!\n")

# ─── Emotion Label → Emoji Mapping ────────────────────────────────────────────
EMOTION_EMOJI = {
    "joy":      "😄",
    "sadness":  "😢",
    "anger":    "😠",
    "fear":     "😨",
    "surprise": "😲",
    "disgust":  "🤢",
    "neutral":  "😐",
}


# ─── Helper: Chunk Long Text ───────────────────────────────────────────────────
def chunk_text(text: str, max_words: int = 180) -> list[str]:
    """
    Break a long piece of text into bite-sized chunks the summarizer can handle.
    Summarization models have a token limit (~1 024 tokens), so anything longer
    needs to be split first. We use word count as a safe proxy for token count.
    """
    words = text.split()
    return [
        " ".join(words[i : i + max_words])
        for i in range(0, len(words), max_words)
    ]


# ─── Step 1: Transcription ────────────────────────────────────────────────────
def transcribe_audio(audio_path: str) -> str:
    """
    Turn the audio file into plain text using OpenAI's Whisper model.

    Whisper is great at handling background noise, accents, and variable audio
    quality. We pass chunk_length_s=30 so it automatically handles recordings
    longer than 30 seconds — no manual splitting needed.
    """
    result = transcriber(audio_path, chunk_length_s=30, batch_size=8)
    return result["text"].strip()


# ─── Step 2: Summarization ────────────────────────────────────────────────────
def summarize_text(text: str) -> str:
    """
    Condense the transcript into the key points using DistilBART.

    For longer transcripts we split the text into chunks, summarize each one,
    then stitch the partial summaries together. This way even a 60-minute
    recording produces a coherent, readable summary.
    """
    word_count = len(text.split())

    # Very short transcripts don't benefit from summarization.
    if word_count < 30:
        return "(Text too short to summarize — showing original)\n\n" + text

    chunks = chunk_text(text, max_words=180)
    partial_summaries = []

    for chunk in chunks:
        # Skip near-empty tail chunks that sometimes appear after splitting.
        if len(chunk.split()) < 10:
            continue
        inputs = _sum_tokenizer(
            chunk, return_tensors="pt", max_length=1024, truncation=True
        )
        if device == 0:
            inputs = {k: v.cuda() for k, v in inputs.items()}
        ids = _sum_model.generate(
            inputs["input_ids"],
            max_length=130,
            min_length=30,
            num_beams=4,
            do_sample=False,
        )
        partial_summaries.append(
            _sum_tokenizer.decode(ids[0], skip_special_tokens=True)
        )

    return " ".join(partial_summaries)


# ─── Step 3: Emotion Detection ────────────────────────────────────────────────
def detect_emotions(text: str) -> str:
    """
    Score the text across 7 emotions and display them as a quick bar chart.

    The emotion model caps out at 512 tokens (~400 words), so we feed it the
    opening of the transcript. In practice the emotional tone of a conversation
    comes through clearly in the first few hundred words.
    """
    # Truncate to ~400 words to stay within the model's token budget.
    truncated = " ".join(text.split()[:400])

    scores = emotion_detector(truncated)[0]   # list of {label, score}
    scores.sort(key=lambda x: x["score"], reverse=True)

    lines = []
    for item in scores:
        label = item["label"]
        pct   = item["score"] * 100
        emoji = EMOTION_EMOJI.get(label, "")
        # Build a simple ASCII bar (each block ≈ 5 %)
        bar   = "█" * int(pct / 5)
        lines.append(f"{emoji} {label:<10} {bar:<20} {pct:5.1f}%")

    return "\n".join(lines)


# ─── Main Pipeline ────────────────────────────────────────────────────────────
def process_audio(audio_file):
    """
    Run the full pipeline: audio → transcript → summary → emotions.

    This is the function wired to the Analyze button. It calls each step in
    order and returns the three result strings to the Gradio UI. Any exception
    is caught and shown as a friendly error message instead of crashing.
    """
    if audio_file is None:
        return (
            "⚠️  Please upload an audio file first.",
            "",
            "",
        )

    try:
        # --- Transcribe ---
        transcript = transcribe_audio(audio_file)
        if not transcript:
            return (
                "⚠️  Transcription returned empty output. "
                "Please check that the file contains audible speech.",
                "",
                "",
            )

        # --- Summarize ---
        summary = summarize_text(transcript)

        # --- Emotions ---
        emotions = detect_emotions(transcript)

        return transcript, summary, emotions

    except Exception as exc:
        # Surface the error message in the UI so users know what went wrong.
        return f"❌ Error: {exc}", "", ""


# ─── Gradio UI ────────────────────────────────────────────────────────────────
with gr.Blocks(
    title="Audio Insight Tool",
) as demo:

    gr.Markdown(
        """
        # 🎙️ Audio Insight Tool
        Upload a **MP3** or **WAV** file to instantly get:
        - 📝 A full **transcription**
        - 📋 A concise **summary**
        - 🎭 An **emotion analysis** of the speech
        """
    )

    with gr.Row():
        # ── Left column: input + button ──
        with gr.Column(scale=1):
            audio_input = gr.Audio(
                type="filepath",
                label="🔊 Upload Audio (MP3 / WAV)",
            )
            analyze_btn = gr.Button("✨ Analyze", variant="primary", size="lg")
            gr.Markdown(
                "_Processing time depends on audio length and hardware. "
                "First run downloads the models (~1 GB total)._"
            )

        # ── Right column: status placeholder while idle ──
        with gr.Column(scale=2):
            transcript_output = gr.Textbox(
                label="📝 Transcription",
                lines=10,
                placeholder="Transcription will appear here…",
            )

    with gr.Row():
        summary_output = gr.Textbox(
            label="📋 Summary",
            lines=5,
            placeholder="Summary will appear here…",
        )
        emotion_output = gr.Textbox(
            label="🎭 Emotion Analysis",
            lines=9,
            placeholder="Emotion scores will appear here…",
        )

    # Wire button click to the processing pipeline.
    analyze_btn.click(
        fn=process_audio,
        inputs=[audio_input],
        outputs=[transcript_output, summary_output, emotion_output],
    )

    gr.Markdown("---")
    gr.Markdown(
        "Built with [Hugging Face Transformers](https://huggingface.co/docs/transformers) "
        "· [Whisper](https://huggingface.co/openai/whisper-small) "
        "· [DistilBART](https://huggingface.co/sshleifer/distilbart-cnn-12-6) "
        "· [Emotion Model](https://huggingface.co/j-hartmann/emotion-english-distilroberta-base)"
    )


# ─── Entry Point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # share=False keeps the app local. Flip it to share=True and Gradio will
    # hand you a public URL that's valid for 72 hours — great for sharing a
    # quick demo without deploying anywhere.
    demo.launch(share=False, theme=gr.themes.Soft())
