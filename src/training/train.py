import os
import logging
import torch
import pandas as pd
from transformers import (
    AutoModelForSeq2SeqLM, 
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    TrainingArguments,
    Trainer
)
from datasets import load_from_disk
import evaluate
from tqdm import tqdm
import json
from datetime import datetime
from config.settings import (
    MODEL_NAME, DATASET_PATH, DATASET_URL, FINE_TUNED_MODEL_PATH, TOKENIZER_PATH,
    BATCH_SIZE, EVAL_BATCH_SIZE, NUM_EPOCHS, WARMUP_STEPS, WEIGHT_DECAY,
    GRADIENT_ACCUMULATION_STEPS, LEARNING_RATE, MAX_INPUT_LENGTH, MAX_TARGET_LENGTH,
    SAVE_STEPS, EVAL_STEPS, LOGGING_STEPS, DEVICE, LOG_LEVEL, LOG_FILE,
    LENGTH_PENALTY, NUM_BEAMS, MAX_GENERATION_LENGTH
)

try:
    from accelerate import __version__ as accelerate_version
    accelerate_major_version = int(accelerate_version.split('.')[0])
    accelerate_minor_version = int(accelerate_version.split('.')[1])
    
    if accelerate_major_version > 0 or accelerate_minor_version >= 21:
        import os
        os.environ["ACCELERATE_USE_FSDP"] = "false"
        os.environ["ACCELERATE_USE_DEEPSPEED"] = "false"
except:
    pass

 
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SummarizationTrainer:
    
    def __init__(self):
        self.device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        self.tokenizer = None
        self.model = None
        self.dataset = None
        self.trainer = None
        
    def setup_model_and_tokenizer(self):
        logger.info(f"Loading base model: {MODEL_NAME}")
        
        import os
        from pathlib import Path
        
        cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
        model_cache_path = Path(cache_dir) / f"models--{MODEL_NAME.replace('/', '--')}"
        
        if model_cache_path.exists():
            logger.info("✅ Model found in cache, loading from local cache...")
        else:
            logger.info("⬇️ Model not in cache, downloading (this may take a few minutes)...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME,
            cache_dir=cache_dir,
            local_files_only=False  
        )
        
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            MODEL_NAME,
            cache_dir=cache_dir,
            local_files_only=False 
        ).to(self.device)
        
        logger.info("Model and tokenizer loaded successfully")
    
    def download_and_prepare_dataset(self):
        logger.info("Preparing SAMSum dataset...")
        
        if os.path.exists(DATASET_PATH):
            logger.info("✅ Dataset found locally, loading from cache...")
        else:
            logger.info("⬇️ Dataset not found, downloading...")
            import urllib.request
            import zipfile
            
            os.makedirs("./data", exist_ok=True)
            zip_path = "./data/summarizer-data.zip"
            
            if os.path.exists(zip_path):
                logger.info("📦 Zip file already exists, extracting...")
            else:
                logger.info(f"⬇️ Downloading from {DATASET_URL}...")
                urllib.request.urlretrieve(DATASET_URL, zip_path)
            
            logger.info("📂 Extracting dataset...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall("./data")
            
            os.remove(zip_path)
            logger.info("✅ Dataset download and extraction completed")
        
        logger.info("📖 Loading dataset...")
        self.dataset = load_from_disk(DATASET_PATH)
        
        split_lengths = [len(self.dataset[split]) for split in self.dataset]
        logger.info(f"📊 Dataset splits: {dict(zip(self.dataset.keys(), split_lengths))}")
        logger.info(f"🔧 Features: {self.dataset['train'].column_names}")
        
        return self.dataset
    
    def preprocess_dataset(self):
        """Preprocess the dataset for training"""
        logger.info("Preprocessing dataset...")
        
        def feature_creation(example_batch):
            """Convert raw text to model inputs"""
            input_encodings = self.tokenizer(
                example_batch['dialogue'], 
                max_length=MAX_INPUT_LENGTH, 
                truncation=True
            )
            
            with self.tokenizer.as_target_tokenizer():
                target_encodings = self.tokenizer(
                    example_batch['summary'], 
                    max_length=MAX_TARGET_LENGTH, 
                    truncation=True
                )
            
            return {
                'input_ids': input_encodings['input_ids'],
                'attention_mask': input_encodings['attention_mask'],
                'labels': target_encodings['input_ids']
            }
        
        logger.info("🏭 PRODUCTION MODE: Using FULL dataset for comprehensive training")
        logger.info(f"📊 Full dataset - Train: {len(self.dataset['train'])}, Val: {len(self.dataset['validation'])}, Test: {len(self.dataset['test'])}")
        logger.info("⏰ Expected training time: 3-4 hours")
        
         
        self.dataset_processed = {split: data.map(feature_creation, batched=True) 
                                 for split, data in self.dataset.items()}
        
        logger.info("Dataset preprocessing completed")
        return self.dataset_processed
    
    def setup_trainer(self):
        """Setup the Hugging Face trainer"""
        logger.info("Setting up trainer...")
        
         
        data_collator = DataCollatorForSeq2Seq(self.tokenizer, model=self.model)
        
         
        training_args = TrainingArguments(
            output_dir='./models/training_checkpoints',
            num_train_epochs=NUM_EPOCHS,
            warmup_steps=WARMUP_STEPS,
            per_device_train_batch_size=BATCH_SIZE,
            per_device_eval_batch_size=EVAL_BATCH_SIZE,
            weight_decay=WEIGHT_DECAY,
            logging_steps=LOGGING_STEPS,
            save_steps=SAVE_STEPS,
            eval_steps=EVAL_STEPS,
            gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
            save_total_limit=1,   
            logging_dir='./logs',
            learning_rate=LEARNING_RATE,
            report_to=[],   
            dataloader_num_workers=0,   
            disable_tqdm=False,   
        )
        
         
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            train_dataset=self.dataset_processed["train"],
             
        )
        
        logger.info("Trainer setup completed")
        return self.trainer
    
    def train(self):
        """Execute the training process"""
        logger.info("Starting training...")
        
         
        training_result = self.trainer.train()
        
         
        logger.info(f"Training completed. Final loss: {training_result.training_loss}")
        
        return training_result
    
    def evaluate(self):
        """Evaluate the model using ROUGE metrics"""
        logger.info("Starting evaluation...")
        
        def generate_batch_sized_chunks(list_of_elements, batch_size):
            """Split the dataset into smaller batches"""
            for i in range(0, len(list_of_elements), batch_size):
                yield list_of_elements[i : i + batch_size]
        
        def calculate_rouge_scores(dataset, model, tokenizer, batch_size=16):
            """Calculate ROUGE scores on test dataset"""
            rouge_metric = evaluate.load('rouge')
            
            article_batches = list(generate_batch_sized_chunks(dataset['dialogue'], batch_size))
            target_batches = list(generate_batch_sized_chunks(dataset['summary'], batch_size))
            
            for article_batch, target_batch in tqdm(
                zip(article_batches, target_batches), 
                total=len(article_batches),
                desc="Evaluating"
            ):
                inputs = tokenizer(
                    article_batch, 
                    max_length=MAX_INPUT_LENGTH, 
                    truncation=True,
                    padding="max_length", 
                    return_tensors="pt"
                )
                
                summaries = model.generate(
                    input_ids=inputs["input_ids"].to(self.device),
                    attention_mask=inputs["attention_mask"].to(self.device),
                    length_penalty=LENGTH_PENALTY, 
                    num_beams=NUM_BEAMS, 
                    max_length=MAX_GENERATION_LENGTH
                )
                
                decoded_summaries = [
                    tokenizer.decode(s, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                    for s in summaries
                ]
                
                decoded_summaries = [d.replace("", " ") for d in decoded_summaries]
                rouge_metric.add_batch(predictions=decoded_summaries, references=target_batch)
            
            return rouge_metric.compute()
        
         
        rouge_scores = calculate_rouge_scores(
            self.dataset['test'], 
            self.trainer.model, 
            self.tokenizer, 
            batch_size=EVAL_BATCH_SIZE
        )
        
         
        rouge_names = ["rouge1", "rouge2", "rougeL", "rougeLsum"]
        rouge_dict = {rn: rouge_scores[rn].mid.fmeasure for rn in rouge_names}
        
         
        results_df = pd.DataFrame([rouge_dict], index=['pegasus-samsum'])
        
         
        logger.info("ROUGE Evaluation Results:")
        logger.info(results_df.to_string())
        
         
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"./logs/evaluation_results_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump({
                'timestamp': timestamp,
                'rouge_scores': rouge_dict,
                'model_name': MODEL_NAME,
                'training_config': {
                    'epochs': NUM_EPOCHS,
                    'batch_size': BATCH_SIZE,
                    'learning_rate': LEARNING_RATE
                }
            }, f, indent=2)
        
        logger.info(f"Evaluation results saved to {results_file}")
        
        return rouge_dict
    
    def save_model(self):
        """Save the fine-tuned model"""
        logger.info("Saving fine-tuned model...")
        
         
        os.makedirs(FINE_TUNED_MODEL_PATH, exist_ok=True)
        os.makedirs(TOKENIZER_PATH, exist_ok=True)
        
         
        self.trainer.model.save_pretrained(FINE_TUNED_MODEL_PATH)
        self.tokenizer.save_pretrained(TOKENIZER_PATH)
        
        logger.info(f"Model saved to {FINE_TUNED_MODEL_PATH}")
        logger.info(f"Tokenizer saved to {TOKENIZER_PATH}")
    
    def run_full_pipeline(self):
        """Run the complete training pipeline"""
        logger.info("Starting full training pipeline...")
        
        try:
             
            self.setup_model_and_tokenizer()
            self.download_and_prepare_dataset()
            self.preprocess_dataset()
            self.setup_trainer()
            
             
            training_result = self.train()
            
             
            rouge_scores = self.evaluate()
            
             
            self.save_model()
            
            logger.info("Training pipeline completed successfully!")
            
            return {
                'training_result': training_result,
                'rouge_scores': rouge_scores,
                'model_path': FINE_TUNED_MODEL_PATH,
                'tokenizer_path': TOKENIZER_PATH
            }
            
        except Exception as e:
            logger.error(f"Training pipeline failed: {str(e)}")
            raise

def main():
    """Main training function"""
    trainer = SummarizationTrainer()
    results = trainer.run_full_pipeline()
    print("Training completed!")
    print(f"ROUGE scores: {results['rouge_scores']}")

if __name__ == "__main__":
    main()
