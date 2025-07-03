# SummarizeIt 🚀

**An end-to-end AI-powered text summarization web application**

Transform long texts and chat conversations into concise, meaningful summaries using a fine-tuned Pegasus transformer trained on the SAMSum dataset. Features automated training pipeline, modern web interface, and containerized deployment.

## 📋 Table of Contents
- [Features](#-features)
- [Quick Start](#-quick-start)
- [Installation](#-installation)
- [Usage](#-usage)
- [Training](#-training)
- [API Documentation](#-api-documentation)
- [Docker Deployment](#-docker-deployment)
- [Project Structure](#-project-structure)
- [Contributing](#-contributing)

## ⭐ Features

### � AI-Powered
- **Fine-tuned Pegasus Model**: Optimized for both chat and article summarization
- **Smart Input Detection**: Automatically detects chat vs. paragraph format
- **Multiple Summary Lengths**: Short, medium, and long summary options
- **ROUGE Evaluation**: Comprehensive model performance metrics

### 🌐 Web Interface
- **Modern UI**: Clean, responsive design with Bootstrap
- **Real-time Processing**: Fast summarization with loading indicators
- **File Upload**: Support for .txt and .md files
- **Copy-to-Clipboard**: Easy result sharing
- **Mobile Responsive**: Works on all devices

### 🔧 Developer Features
- **Modular Architecture**: Separate training, model, and app components
- **Automated Pipeline**: One-command training and deployment
- **Docker Support**: Fully containerized with docker-compose
- **API First**: RESTful API with OpenAPI documentation
- **Configuration Management**: Centralized settings and environment variables

### 📊 Input Support
- **Text Types**: Articles, blog posts, news, academic papers
- **Chat Formats**: WhatsApp, Telegram, Slack, Teams conversations
- **File Formats**: .txt, .md files up to 10MB
- **Batch Processing**: Multiple text summarization via API

## 🚀 Quick Start

### Option 1: One-Command Setup
```bash
# Clone and setup everything automatically
git clone <repository-url>
cd SummarizeIt
python setup_and_run.py
```

### Option 2: Manual Setup
```bash
# 1. Clone repository
git clone <repository-url>
cd SummarizeIt

# 2. Create virtual environment
python -m venv venv

# 3. Activate virtual environment
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Setup project
python main.py setup

# 6. Start web app (uses base model)
python main.py serve
```

### Option 3: Docker
```bash
# Clone and run with Docker
git clone <repository-url>
cd SummarizeIt
docker-compose up --build
```

**🌐 Access the web interface at: http://localhost:8000**

## � Installation

### Prerequisites
- Python 3.8+ 
- 8GB+ RAM (for model training)
- CUDA GPU (optional, for faster training)
- Docker (optional, for containerized deployment)

### Dependencies
```bash
# Core ML/NLP
transformers[sentencepiece]==4.36.2
torch>=2.0.0
datasets==2.14.7

# Web framework
fastapi==0.104.1
uvicorn[standard]==0.24.0

# Data processing
pandas>=2.0.0
nltk>=3.8.1
rouge-score==0.1.2
```

## 🎯 Usage

### Web Interface
1. **Open**: Navigate to http://localhost:8000
2. **Choose Mode**: Text input or file upload
3. **Enter Content**: Paste text or upload file
4. **Configure**: Select summary length and content type
5. **Generate**: Click "Generate Summary"
6. **Results**: View summary with compression statistics

### Command Line Interface
```bash
# Setup environment
python main.py setup

# Check dependencies
python main.py check

# Train model (optional - takes 2-4 hours)
python main.py train

# Evaluate model
python main.py evaluate

# Start web server
python main.py serve --host 0.0.0.0 --port 8000 --reload
```

### API Usage
```python
import requests

# Summarize text
response = requests.post("http://localhost:8000/summarize", json={
    "text": "Your long text here...",
    "summary_length": "medium",
    "input_type": "auto"
})

result = response.json()
print(result["summary"])
```

## 🧠 Training

### Quick Training (Recommended)
```bash
# Train with default settings
python main.py train
```

### Custom Training
```python
from src.training.train import SummarizationTrainer

trainer = SummarizationTrainer()
# Modify config/settings.py for custom parameters
results = trainer.run_full_pipeline()
```

### Training Process
1. **Download Dataset**: Automatic SAMSum dataset download
2. **Preprocessing**: Tokenization and feature creation
3. **Training**: 4 epochs with validation
4. **Evaluation**: ROUGE score calculation
5. **Model Saving**: Automated model and tokenizer saving

### Training Configuration
Edit `config/settings.py`:
```python
NUM_EPOCHS = 4              # Training epochs
BATCH_SIZE = 2              # Batch size
LEARNING_RATE = 5e-5        # Learning rate
MAX_INPUT_LENGTH = 1024     # Max input tokens
MAX_TARGET_LENGTH = 128     # Max summary tokens
```

## 📚 API Documentation

### Endpoints

#### `POST /summarize`
Summarize single text
```json
{
  "text": "Your text here...",
  "summary_length": "medium",  // short, medium, long
  "input_type": "auto"         // auto, chat, paragraph
}
```

#### `POST /summarize/batch`
Summarize multiple texts
```json
{
  "texts": ["Text 1", "Text 2"],
  "summary_length": "medium",
  "input_type": "auto"
}
```

#### `POST /summarize/file`
Upload and summarize file
```bash
curl -X POST "http://localhost:8000/summarize/file" \
  -F "file=@document.txt" \
  -F "summary_length=medium" \
  -F "input_type=auto"
```

#### `GET /health`
API health check
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_type": "fine-tuned"
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

## 🐳 Docker Deployment

### Development
```bash
# Build and run
docker-compose up --build

# Run in background
docker-compose up -d

# View logs
docker-compose logs -f
```

### Production
```bash
# Use production profile with nginx
docker-compose --profile production up -d

# Scale application
docker-compose up --scale summarize-app=3
```

### Environment Variables
```env
PYTHONPATH=/app
LOG_LEVEL=INFO
MODEL_NAME=google/pegasus-cnn_dailymail
DEVICE=cuda  # or cpu
```

## 📁 Project Structure

```
SummarizeIt/
├── 📁 src/
│   ├── 📁 app/              # Web application
│   │   ├── main.py          # FastAPI app
│   │   ├── 📁 templates/    # HTML templates
│   │   └── 📁 static/       # CSS, JS, images
│   ├── 📁 model/            # Model handling
│   │   └── summarizer.py    # Model wrapper
│   └── 📁 training/         # Training pipeline
│       └── train.py         # Training logic
├── 📁 config/               # Configuration
│   └── settings.py          # App settings
├── 📁 models/               # Saved models
├── 📁 data/                 # Datasets
├── 📁 logs/                 # Log files
├── 📁 Notebooks/            # Jupyter notebooks
├── main.py                  # CLI entry point
├── setup_and_run.py         # Quick setup script
├── requirements.txt         # Dependencies
├── Dockerfile               # Docker configuration
├── docker-compose.yml       # Docker Compose
└── README.md               # This file
```

## 🔧 Configuration

### Model Configuration
- **Base Model**: `google/pegasus-cnn_dailymail`
- **Fine-tuned Path**: `./models/pegasus-samsum-model`
- **Input Length**: 1024 tokens max
- **Output Length**: 128 tokens max

### Summary Lengths
- **Short**: 10-50 words (1-2 sentences)
- **Medium**: 30-100 words (3-5 sentences)  
- **Long**: 50-150 words (detailed overview)

### Performance
- **Base Model**: ~2-3 seconds per summary
- **Fine-tuned Model**: ~2-3 seconds per summary
- **Batch Processing**: ~1 second per text in batch
- **Memory Usage**: ~2GB GPU / ~4GB CPU

## 🤝 Contributing

### Development Setup
```bash
# Clone repository
git clone <repository-url>
cd SummarizeIt

# Setup development environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt

# Install development dependencies
pip install pytest black flake8

# Run tests
pytest

# Format code
black .

# Lint code
flake8 .
```

### Code Structure
- **`src/model/`**: Model loading and inference
- **`src/training/`**: Training pipeline and evaluation
- **`src/app/`**: Web application and API
- **`config/`**: Configuration and settings

### Adding Features
1. **New Model**: Add to `src/model/`
2. **New Training**: Extend `src/training/`
3. **New API Endpoint**: Add to `src/app/main.py`
4. **New UI Feature**: Update templates and static files

## 📊 Model Performance

### ROUGE Scores (Fine-tuned vs Base)
| Metric | Base Model | Fine-tuned | Improvement |
|--------|------------|------------|-------------|
| ROUGE-1 | 0.42 | 0.47 | +11.9% |
| ROUGE-2 | 0.19 | 0.23 | +21.1% |
| ROUGE-L | 0.34 | 0.39 | +14.7% |

### Use Case Performance
- **Chat Summarization**: Excellent (fine-tuned on SAMSum)
- **News Articles**: Very Good (pre-trained on CNN/DailyMail)
- **Academic Papers**: Good (general language understanding)
- **Meeting Notes**: Very Good (conversation understanding)

## 🚨 Troubleshooting

### Common Issues

**Model Loading Errors**
```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# Use CPU if CUDA issues
export DEVICE=cpu
python main.py serve
```

**Memory Issues**
```bash
# Reduce batch size in config/settings.py
BATCH_SIZE = 1
EVAL_BATCH_SIZE = 1
```

**Port Already in Use**
```bash
# Use different port
python main.py serve --port 8001
```

**Dependencies Missing**
```bash
# Reinstall requirements
pip install -r requirements.txt --upgrade
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Hugging Face Transformers**: Model implementation
- **Google**: Pegasus architecture
- **SAMSum Dataset**: Fine-tuning data
- **FastAPI**: Web framework
- **Bootstrap**: UI components

---

**🚀 Ready to summarize? Start with:** `python setup_and_run.py`


