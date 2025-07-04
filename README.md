# SummarizeIt 🚀

<div align="center">

**Transform lengthy texts into concise, meaningful summaries with AI**

[![Live Demo](https://img.shields.io/badge/🚀_Live_Demo-Hugging_Face-yellow.svg)](https://huggingface.co/spaces/Ananthakr1shnan/summarize-it)
[![Model](https://img.shields.io/badge/🤖_Model-Hugging_Face-blue.svg)](https://huggingface.co/Ananthakr1shnan/pegasus-samsum-finetuned)
[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

*An end-to-end AI-powered text summarization web application built with fine-tuned Pegasus transformer*

**Tech Stack:** Python • FastAPI • Transformers • PyTorch • Bootstrap • Docker

[🎯 **Try it Live**](https://huggingface.co/spaces/Ananthakr1shnan/summarize-it) • [🤖 **Model**](https://huggingface.co/Ananthakr1shnan/pegasus-samsum-finetuned) • [🐛 **Report Bug**](https://github.com/ananthakr1shnan/SummarizeIt/issues)

</div>

---

## ✨ Features

- **🧠 Fine-tuned Pegasus Model** - Optimized for both chat and article summarization using SAMSum dataset
- **🤗 Hugging Face Integration** - Model deployed on Hugging Face Hub, app hosted on Spaces
- **🎯 Smart Input Detection** - Automatically detects chat vs. paragraph format
- **📏 Multiple Summary Lengths** - Short, medium, and long summary options
- **🌐 Modern Web Interface** - Clean, responsive design with real-time processing
- **📁 File Upload Support** - Handle .txt and .md files up to 10MB
- **🔌 RESTful API** - Complete API with OpenAPI documentation
- **📊 ROUGE Evaluation** - Comprehensive model performance metrics

---

## 🔄 How It Works

### 📊 **Complete Workflow**

```mermaid
graph TD
    A[📚 SAMSum Dataset] --> B[🔧 Data Preprocessing]
    B --> C[🧠 Fine-tune Pegasus Model]
    C --> D[📈 Model Evaluation]
    D --> E[💾 Save Fine-tuned Model]
    E --> F[🤗 Upload to Hugging Face Hub]
    F --> G[🚀 Deploy FastAPI App]
    G --> H[🌐 Hugging Face Spaces]
    
    I[👤 User Input] --> J{📝 Input Type Detection}
    J -->|Chat| K[💬 Chat Processing]
    J -->|Article| L[📄 Article Processing]
    K --> M[🤖 Pegasus Model]
    L --> M
    M --> N[📋 Generate Summary]
    N --> O[📊 Calculate Metrics]
    O --> P[✅ Return Results]
```

### 🎯 **Processing Pipeline**

1. **📥 Input Processing**
   - Detect input type (chat conversation vs. article)
   - Clean and preprocess text
   - Handle file uploads (.txt, .md)

2. **🧠 AI Summarization**
   - Load fine-tuned Pegasus model from Hugging Face Hub
   - Tokenize input text (max 1024 tokens)
   - Generate summary based on selected length
   - Apply post-processing filters

3. **📊 Output Generation**
   - Calculate compression ratio
   - Compute summary statistics
   - Format response with metadata
   - Return JSON response or web interface

### 🔧 **Model Training Pipeline**

```mermaid
graph LR
    A[📚 SAMSum Dataset<br/>16k+ conversations] --> B[🔧 Tokenization<br/>Max 1024 tokens]
    B --> C[🎯 Fine-tuning<br/>4 epochs]
    C --> D[📈 ROUGE Evaluation<br/>R-1, R-2, R-L]
    D --> E[💾 Model Export<br/>HuggingFace format]
```

**Training Stats:**
- **Dataset:** 16,000+ chat conversations
- **Training Time:** 2-4 hours (GPU) / 8-12 hours (CPU)
- **Model Size:** ~2.3GB
- **Performance:** 11.9% improvement in ROUGE-1 score

---

## 🚀 Quick Start

### 🎯 **Try Online** (No Installation Required)
[![Open in Hugging Face](https://img.shields.io/badge/🤗_Open_in-Hugging_Face_Spaces-yellow.svg?style=for-the-badge)](https://huggingface.co/spaces/Ananthakr1shnan/summarize-it)

### 💻 **Local Setup**

**Option 1: Manual Setup**
```bash
git clone https://github.com/ananthakr1shnan/SummarizeIt.git
cd SummarizeIt
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
python main.py setup
python main.py serve
```

**Option 2: Docker**
```bash
git clone https://github.com/ananthakr1shnan/SummarizeIt.git
cd SummarizeIt
docker-compose up --build
```


**🌐 Access the app at:** `http://localhost:8000`

---

## 📋 Requirements

**Tech Stack:**
- **Backend:** Python 3.11+, FastAPI, Uvicorn
- **AI/ML:** Transformers, PyTorch, Datasets
- **Frontend:** Bootstrap, HTML/CSS/JavaScript
- **Deployment:** Docker, Hugging Face Spaces
- **Model:** Fine-tuned Pegasus (google/pegasus-cnn_dailymail)

**System Requirements:**
- 8GB+ RAM (for model training)
- CUDA GPU (optional, for faster training)
- Docker (optional)

---

## 🎯 Usage

### Web Interface
1. Open `http://localhost:8000`
2. Paste text or upload a file
3. Select summary length and content type
4. Click "Generate Summary"
5. Copy and share your summary

### API Usage
```python
import requests

response = requests.post("http://localhost:8000/summarize", json={
    "text": "Your long text here...",
    "summary_length": "medium",
    "input_type": "auto"
})

result = response.json()
print(result["summary"])
```

### CLI Commands
```bash
python main.py setup      # Setup environment
python main.py train      # Train model (2-4 hours)
python main.py serve      # Start web server
python main.py evaluate   # Evaluate model
```

---

## 🧠 Model Performance

### ROUGE Scores (Fine-tuned vs Base)
| Metric | Base Model | Fine-tuned | Improvement |
|--------|------------|------------|-------------|
| ROUGE-1 | 0.42 | **0.47** | +11.9% |
| ROUGE-2 | 0.19 | **0.23** | +21.1% |
| ROUGE-L | 0.34 | **0.39** | +14.7% |

### Performance Stats
- **Processing Speed**: ~2-3 seconds per summary
- **Memory Usage**: ~2GB GPU / ~4GB CPU
- **Batch Processing**: ~1 second per text in batch

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/summarize` | Summarize single text |
| `POST` | `/summarize/batch` | Summarize multiple texts |
| `POST` | `/summarize/file` | Upload and summarize file |
| `GET` | `/health` | API health check |
| `GET` | `/docs` | Interactive API documentation |

### Request Format
```json
{
  "text": "Your text here...",
  "summary_length": "medium",  // "short", "medium", "long"
  "input_type": "auto"         // "auto", "chat", "paragraph"
}
```

### Response Format
```json
{
  "summary": "Generated summary text...",
  "original_length": 450,
  "summary_length": 89,
  "compression_ratio": 0.2,
  "summary_type": "medium",
  "input_type": "paragraph"
}
```
---

## 🎓 Training

Train your own model with custom data:

```bash
# Quick training with default settings
python main.py train

# Custom configuration in config/settings.py
NUM_EPOCHS = 4 (For high perfomance, train with 1 epoch in minimum hardware conditions)
BATCH_SIZE = 2
LEARNING_RATE = 5e-5
```

The training process includes:
1. Download SAMSum dataset
2. Preprocessing and tokenization
3. Model training 
4. ROUGE evaluation
5. Model saving

---

## 🚀 Deployment

### 🤗 **Hugging Face Deployment**
- **Live App:** Deployed on [Hugging Face Spaces](https://huggingface.co/spaces/Ananthakr1shnan/summarize-it)
- **Model Hub:** Fine-tuned model available at [Hugging Face Hub](https://huggingface.co/Ananthakr1shnan/pegasus-samsum-finetuned)
- **Zero Setup:** No installation required, just click and use!

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

