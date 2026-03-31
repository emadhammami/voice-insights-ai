# Voice Insights AI

**Author:** Emad Hammami

A local NLP tool that takes an audio recording and produces three outputs in one pass:

1. **Transcript** — full text of what was said (Whisper small)
2. **Summary** — concise version of the key points (DistilBART)
3. **Emotion breakdown** — per-emotion confidence scores across 7 categories (DistilRoBERTa)

The motivation was practical: I needed a way to extract information from long meeting and lecture recordings without listening to them in full. The tool runs entirely on your machine — no external APIs, no data sent anywhere.

---

## Models

| Task | Model | Why this one |
|------|-------|-------------|
| Speech recognition | `openai/whisper-small` | Strong multilingual performance, handles noisy recordings well, fast enough on CPU |
| Summarisation | `sshleifer/distilbart-cnn-12-6` | Distilled version of BART, good abstractive quality at a fraction of the compute |
| Emotion classification | `j-hartmann/emotion-english-distilroberta-base` | Covers 7 Ekman-based categories, well-documented fine-tuning methodology |

All weights are downloaded automatically from Hugging Face on first run (~900 MB total).

---

## Project layout

```
voice-insights-ai/
├── app.py            # pipeline logic + Gradio UI
├── requirements.txt  # dependencies
└── README.md
```

---

## Setup

```bash
git clone https://github.com/emadhammami/voice-insights-ai.git
cd voice-insights-ai

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

**GPU (optional):** install the CUDA-enabled PyTorch build before the rest:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

**ffmpeg:** handled automatically. The `imageio-ffmpeg` package ships a static binary; the app registers it at startup so no system install is needed.

---

## Running

```bash
python app.py
```

Opens at `http://127.0.0.1:7860`. First run is slower — models need to download. After that they load from the local Hugging Face cache.

---

## Usage

1. Upload an MP3 or WAV file.
2. Click **Analyse**.
3. Read the transcript, summary, and emotion scores.

Sample output for a short interview recording:

**Transcript**
```
Today we want to discuss the role of explainability in modern machine learning systems,
particularly in high-stakes domains like healthcare and criminal justice...
```

**Summary**
```
Explainability in ML is critical in high-stakes domains. Researchers are working on
methods that make model decisions interpretable without sacrificing accuracy.
```

**Emotions**
```
😐 neutral     ████████████████     80.2%
😄 joy         ██                   11.4%
😢 sadness     █                     5.1%
😠 anger                             2.0%
😨 fear                              0.9%
😲 surprise                          0.3%
🤢 disgust                           0.1%
```

---

## Deploying on Hugging Face Spaces

1. Create a new Space → SDK: **Gradio**
2. Push this repo (or upload the three files manually)
3. The Space installs dependencies and starts automatically

CPU tier works fine. Upgrade to a GPU Space for significantly faster transcription on long files.

---

## Known limitations

- Emotion model is English-only and capped at 512 tokens; only the first ~400 words of the transcript are scored
- Summarisation quality degrades on highly technical or domain-specific speech
- Very short clips (under 5 seconds) often produce empty or unreliable transcripts

---

## Possible extensions

- Speaker diarisation with `pyannote-audio`
- Keyword / topic extraction via zero-shot classification
- Export to PDF or structured JSON
- Multilingual emotion detection
- Streaming transcription with live word-by-word output

---

## License

MIT
