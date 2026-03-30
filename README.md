# 🎙️ Voice Insights AI

Ever sat through a long meeting recording or lecture and wished someone would just tell you what was said, what it meant, and how people felt about it? That's exactly what this tool does.

Drop in an **MP3 or WAV** file and you'll walk away with:

| Step | What happens | Model used |
|------|-------------|------------|
| 1 | **Transcription** | OpenAI Whisper (small) |
| 2 | **Summarization** | DistilBART CNN 12-6 |
| 3 | **Emotion Detection** | j-hartmann/emotion-english-distilroberta-base |

Everything runs locally on your machine (or in a Hugging Face Space). No API keys, no subscriptions, no data leaving your computer.

Built with [Hugging Face Transformers](https://huggingface.co/docs/transformers) and [Gradio](https://gradio.app/).

---

## ✨ Features

- 🎤 **Reliable Speech-to-Text** — Whisper handles noisy recordings, strong accents, and files of any length without you lifting a finger.
- 📋 **Smart Summarization** — No more reading through walls of text. Long transcripts are chunked automatically and condensed into a short, readable summary.
- 🎭 **Emotion Analysis** — See at a glance whether a conversation was mostly upbeat, tense, or neutral across 7 emotion categories.
- 🖥️ **Clean Web UI** — A minimal Gradio interface that anyone can use without any technical knowledge.
- ⚡ **GPU / CPU auto-detection** — Uses your GPU when it finds one, gracefully falls back to CPU otherwise, zero config needed.
- ☁️ **One-click Hugging Face Spaces deploy** — Push the folder, pick Gradio as the SDK, and you're live.

---

## 🗂️ Project Structure

```
voice-insights-ai/
├── app.py            # Everything lives here: models, pipeline logic, and the Gradio UI
├── requirements.txt  # pip dependencies
└── README.md         # You're reading it
```

---

## 🚀 Installation

### Prerequisites

- Python 3.10 or later
- pip

### 1. Clone the repo

```bash
git clone https://github.com/emadhammami/voice-insights-ai.git
cd voice-insights-ai
```

### 2. (Optional) Create a virtual environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **GPU users:** Grab the CUDA-enabled PyTorch build first so you get the speed boost:
> ```bash
> pip install torch --index-url https://download.pytorch.org/whl/cu121
> ```
> Then `pip install -r requirements.txt` as usual.

---

## ▶️ Running Locally

```bash
python app.py
```

Gradio will print a local URL (usually `http://127.0.0.1:7860`). Open it in your browser.

> The **first run** will download all three models (about 900 MB combined). Grab a coffee — it only happens once. After that, everything loads from your local cache.

To create a **public shareable link** (expires after 72 hours), change the last line of `app.py`:

```python
demo.launch(share=True)
```

---

## 📖 Example Usage

1. Open `http://127.0.0.1:7860` in your browser after starting the app.
2. Click the upload area and pick any MP3 or WAV file from your computer.
3. Hit the big **✨ Analyze** button and wait a moment.
4. That's it — your transcript, summary, and emotion scores will appear side by side.

### Sample output

**Transcription**
```
Welcome everyone. Today we're going to talk about the importance of mental health
in the workplace and how small daily habits can make a significant difference…
```

**Summary**
```
Mental health in the workplace is influenced by small daily habits. Employers and
employees alike benefit from simple routines that reduce stress and improve focus.
```

**Emotion Analysis**
```
😄 joy        ████████████████     78.3%
😐 neutral    ████                 18.1%
😢 sadness    █                     2.4%
😠 anger                            0.7%
😨 fear                             0.4%
😲 surprise                         0.1%
🤢 disgust                          0.0%
```

---

## 🖼️ Screenshots

> Screenshots coming soon! Run the app locally or check the live Hugging Face Space once it's deployed.

<!-- Feel free to add your own screenshots here: drag an image into this file on GitHub -->

---

## ☁️ Deploy on Hugging Face Spaces

Want to share it with the world without any infra headaches? Hugging Face Spaces makes it trivial:

1. Go to [huggingface.co/spaces](https://huggingface.co/spaces) and create a new Space.
2. Pick **Gradio** as the SDK.
3. Push this repo (or upload `app.py`, `requirements.txt`, and `README.md` manually).
4. Sit back — the Space installs the dependencies and boots up on its own.

CPU Spaces are free and perfectly usable. If you process a lot of audio, upgrading to a GPU Space will cut inference time significantly.

---

## 🔧 Model Summary

| Purpose | Model | Size |
|---------|-------|------|
| Transcription | [`openai/whisper-small`](https://huggingface.co/openai/whisper-small) | ~244 MB |
| Summarization | [`sshleifer/distilbart-cnn-12-6`](https://huggingface.co/sshleifer/distilbart-cnn-12-6) | ~306 MB |
| Emotion Detection | [`j-hartmann/emotion-english-distilroberta-base`](https://huggingface.co/j-hartmann/emotion-english-distilroberta-base) | ~329 MB |

All models are downloaded automatically from the Hugging Face Hub on first use.

---

## 🛣️ What's Next

There's a lot of room to grow here. Some ideas I'd love to tackle:

- [ ] **Language picker** — let users tell Whisper which language to expect instead of always guessing.
- [ ] **Better Whisper model** — `whisper-large-v3` is noticeably more accurate, especially on accented speech.
- [ ] **Speaker diarization** — figure out *who* is talking when, using `pyannote-audio`.
- [ ] **Topic / keyword extraction** — pull out the main themes automatically with a zero-shot classifier.
- [ ] **Export button** — download your transcript and summary as a `.txt` or `.pdf` file.
- [ ] **Batch mode** — drop in a whole folder of recordings and let it chew through them.
- [ ] **Multilingual emotions** — the current model is English-only; supporting other languages would be huge.
- [ ] **Live streaming** — show words appearing in real time as the audio plays, rather than waiting for the full result.

PRs and ideas are very welcome — open an issue or fork away!

---

## 🤝 Contributing

Found a bug? Have a feature idea? Open an issue or submit a pull request — contributions of all sizes are appreciated.

## 📄 License

MIT — use it, modify it, ship it. Attribution appreciated but not required.
