<div align="center">
    <h1>📣 Real-Time TTS with Memory Caching</h1>
    <p>
        A practical extension and demonstration of the <b><em>Spark-TTS</em></b> system.<br>
        Integrates FAISS-based vector search and Redis for audio reuse in a real-time TTS pipeline.
    </p>
</div>

---
## 📝 Project Info

- **Course**: Machine Learning Programming (AIDI 1002)
- **Project Title**: Real-Time Text-to-Speech with Intelligent Caching using Spark-TTS + FAISS
- **Team Members**:
  - Asilbek Turgunboev
  - Monica Priya

## ✨ Project Summary

This repository enhances Spark-TTS by introducing a **semantic memory caching system** using **FAISS** and **Redis** to detect previously generated TTS outputs and avoid redundant audio synthesis.

- 🔁 Reuses audio segments if similar text has already been spoken.
- 🧠 Leverages semantic similarity with BERT embeddings.
- 🔊 Segments long input into meaningful chunks (sentence or logical units).
- 🧪 Includes full unit tests for storage, search, and caching behavior.

---

## 📐 System Components

- `Spark-TTS`: Main TTS model for generating waveform.
- `FAISS`: Efficient vector similarity search.
- `Redis`: Key-value store for fast lookup and metadata.
- `NLTK`: Used for segmenting user input into logical parts.
- `Gradio`: Provides interactive UI to test inputs and playback audio.

---

## 🧪 How It Works

```plaintext
📥 User Input Text
   ↓
🪄 Text Segmentation
   - Uses NLTK to break the input into logical sentences or chunks.
   ↓
🔍 For Each Segment:
   → [1] Generate BERT Embedding (768-dim)
   → [2] Search Similar Embedding in FAISS
       ↳ If match found (distance < 0.1): ✅ Use Cached Audio
       ↳ If no match: ❌ Generate New Audio via Spark-TTS
                          ↓
                     🎙️ Save Audio + Text to:
                          - Redis (text, audio path, timestamp)
                          - FAISS (embedding)
                          - Index Map (session ID)
   ↓
🖨️ Log Result
   - Printed logs indicate which segments were cached or generated
   ↓
🔊 Output
   - Last audio segment is returned for playback in UI

```

Each segment is checked individually, and processed accordingly, with logs printing:

- ✅ **Cached part** if already stored
- 🎙️ **Generated part** if new audio is created

---

## 📂 Folder Structure

```
├── realtime.py                  # Gradio app with TTS logic
├── segmentation_and_memory/
│   ├── memory_management.py     # FAISS + Redis implementation
│   └── segmentation.py          # Sentence splitting using nltk
├── test_memory.py               # Test cases for logic verification
├── temp_audio/                  # Folder for storing .wav files
├── pretrained_models/           # Spark-TTS checkpoint
└── README.md
```

---

## ▶️ Usage Instructions

1. **Install requirements**
```bash
pip install -r requirements.txt
```

2. **Download Spark-TTS model**
```python
from huggingface_hub import snapshot_download
snapshot_download("SparkAudio/Spark-TTS-0.5B", local_dir="pretrained_models/Spark-TTS-0.5B")
```

3. **Run Gradio UI**
```bash
python realtime.py
```

4. **Run tests**
```bash
python test_memory.py
```

---

## 🧪 Test Coverage

Tested scenarios include:

- ✅ Single segment cache reuse
- 📚 Multiple sentence processing
- 📦 Bulk session storage (10+ entries)
- 🔍 Threshold-based FAISS matching
- 🧹 Cache clearing with `clear_cache()`
- 🎛 Logging generated/cached segments individually

---

## 📊 Example Output (Logs)

```plaintext
INFO - Generated embedding shape: (1, 768)
INFO - Match found: session_0, text: 'hello', distance: 0.0
INFO - Returning cached audio: temp_audio/temp_20250409230446.wav
INFO - Processed text parts:
INFO - Cached part: 'hello'
INFO - Generated part: 'this is a new segment'
```

---

## 📚 Based On

This project builds upon and uses the inference model from:

**Spark-TTS**  
*Spark-TTS: An Efficient LLM-Based Text-to-Speech Model with Single-Stream Decoupled Speech Tokens*  
[🔗 ArXiv Paper](https://arxiv.org/abs/2503.01710)  
[🔗 GitHub Repo](https://github.com/SparkAudio/Spark-TTS)  
[🔗 Model on Hugging Face](https://huggingface.co/SparkAudio/Spark-TTS-0.5B)

```
@misc{wang2025sparktts,
      title={Spark-TTS: An Efficient LLM-Based Text-to-Speech Model with Single-Stream Decoupled Speech Tokens},
      author={Xinsheng Wang et al.},
      year={2025},
      eprint={2503.01710},
      archivePrefix={arXiv},
      primaryClass={cs.SD}
}
```

---

## ⚠️ Ethical Use Disclaimer

This project is strictly for educational and research use. Please do not misuse voice cloning or text-to-speech technology for impersonation, deepfakes, or violating privacy.

---
