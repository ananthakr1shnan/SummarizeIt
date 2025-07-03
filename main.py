import argparse
import sys
import os
import logging
from pathlib import Path

 
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from src.training.train import SummarizationTrainer
from src.model.summarizer import SummarizationModel
from config.settings import *

 
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("./logs/app.log", encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

def train_model():
    logger.info("Starting model training...")
    
    trainer = SummarizationTrainer()
    results = trainer.run_full_pipeline()
    
    logger.info("Training completed successfully!")
    logger.info(f"ROUGE scores: {results['rouge_scores']}")
    logger.info(f"Model saved to: {results['model_path']}")
    
    return results

def evaluate_model(model_path=None, tokenizer_path=None):
    """Evaluate the model on sample text"""
    logger.info("Loading model for evaluation...")
    
     
    model = SummarizationModel(model_path, tokenizer_path)
    
     
    sample_texts = [
        {
            "text": """
            John: Hey Sarah, how was your day at work?
            Sarah: Pretty good! Had a big presentation today.
            John: How did it go?
            Sarah: Really well actually. The client loved our proposal.
            John: That's awesome! We should celebrate.
            Sarah: Definitely! Dinner tomorrow?
            John: Sounds perfect. I'll make reservations.
            """,
            "type": "chat"
        },
        {
            "text": """
            Artificial intelligence has made significant strides in recent years, particularly in the field of natural language processing. Large language models like GPT and BERT have revolutionized how we approach text understanding and generation. These models can perform a wide variety of tasks including translation, summarization, question answering, and creative writing. The training process involves feeding massive amounts of text data to neural networks, allowing them to learn patterns in human language. As these technologies continue to evolve, they promise to transform industries from healthcare to education, making information more accessible and communication more efficient.
            """,
            "type": "paragraph"
        }
    ]
    
    logger.info("Evaluating on sample texts...")
    
    for i, sample in enumerate(sample_texts, 1):
        logger.info(f"\n--- Sample {i} ({sample['type']}) ---")
        logger.info(f"Original text ({len(sample['text'].split())} words):")
        logger.info(sample['text'][:200] + "..." if len(sample['text']) > 200 else sample['text'])
        
         
        for length in ['short', 'medium', 'long']:
            result = model.summarize_text(sample['text'], length)
            logger.info(f"\n{length.capitalize()} summary:")
            logger.info(f"Summary: {result['summary']}")
            logger.info(f"Compression: {result['compression_ratio']}x")

def serve_app(host="0.0.0.0", port=8000, reload=False):
    logger.info(f"Starting web server on {host}:{port}")
    
     
    if host == "0.0.0.0":
        logger.info("🌐 Access the application at:")
        logger.info(f"   Local:    http://localhost:{port}")
        logger.info(f"   Network:  http://127.0.0.1:{port}")
        logger.info("📝 Note: Use 'localhost' or '127.0.0.1' in your browser, NOT '0.0.0.0'")
    else:
        logger.info(f"🌐 Access the application at: http://{host}:{port}")
    
    try:
        import uvicorn
        uvicorn.run(
            "src.app.main:app",
            host=host,
            port=port,
            reload=reload,
            log_level="info"
        )
    except ImportError:
        logger.error("uvicorn not installed. Install with: pip install uvicorn")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        logger.info("💡 Try using a different port with: python main.py serve --port 8001")
        sys.exit(1)

def setup_environment():
    logger.info("Setting up project environment...")
    
     
    directories = [
        'models',
        'data', 
        'logs',
        'src/app/static',
        'src/app/templates'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Created directory: {directory}")
    
     
    venv_detected = False
    venv_type = ""
    
     
    if 'CONDA_DEFAULT_ENV' in os.environ:
        venv_detected = True
        venv_type = f"conda ({os.environ['CONDA_DEFAULT_ENV']})"
     
    elif hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        venv_detected = True
        venv_type = "virtualenv/venv"
     
    elif 'PIPENV_ACTIVE' in os.environ:
        venv_detected = True
        venv_type = "pipenv"
    
    if venv_detected:
        logger.info(f"Virtual environment detected: {venv_type}")
    else:
        logger.warning("No virtual environment detected. Consider using one.")
    
    logger.info("Environment setup completed!")

def check_dependencies():
    required_packages = [
        'transformers',
        'torch',
        'datasets',
        'fastapi',
        'uvicorn'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        logger.error(f"Missing required packages: {missing_packages}")
        logger.error("Install with: pip install -r requirements.txt")
        return False
    
    logger.info("All required packages are installed")
    return True

def main():
    parser = argparse.ArgumentParser(description="SummarizeIt - Text Summarization Pipeline")
    parser.add_argument(
        "command",
        choices=['setup', 'train', 'evaluate', 'serve', 'check', 'test'],
        help="Command to execute"
    )
    parser.add_argument(
        "--model-path",
        help="Path to model for evaluation/serving"
    )
    parser.add_argument(
        "--tokenizer-path", 
        help="Path to tokenizer for evaluation/serving"
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host for serving (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port for serving (default: 8000)"
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development"
    )
    
    args = parser.parse_args()
    
    if args.command == 'setup':
        setup_environment()
        
    elif args.command == 'check':
        if check_dependencies():
            logger.info("✅ All checks passed!")
        else:
            logger.error("❌ Some checks failed!")
            sys.exit(1)
    
    elif args.command == 'test':
         
        logger.info("🧪 Testing model loading...")
        try:
            model = SummarizationModel()
            test_text = "John: Hi! Sarah: Hello! How are you? John: Good, thanks!"
            result = model.summarize_text(test_text, "short")
            logger.info(f"✅ Model test successful!")
            logger.info(f"📝 Test summary: {result['summary']}")
        except Exception as e:
            logger.error(f"❌ Model test failed: {e}")
            logger.info("💡 Make sure you've extracted the trained model files to the models/ folder")
            sys.exit(1)
            
    elif args.command == 'train':
        if not check_dependencies():
            sys.exit(1)
        train_model()
        
    elif args.command == 'evaluate':
        if not check_dependencies():
            sys.exit(1)
        evaluate_model(args.model_path, args.tokenizer_path)
        
    elif args.command == 'serve':
        if not check_dependencies():
            sys.exit(1)
        serve_app(args.host, args.port, args.reload)

if __name__ == "__main__":
    main()
