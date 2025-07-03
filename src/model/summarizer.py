import torch
import logging
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
from typing import Dict, List, Optional, Union
import os
from config.settings import *

logger = logging.getLogger(__name__)

class SummarizationModel:
    """Handler for the Pegasus summarization model"""
    
    def __init__(self, model_path: Optional[str] = None, tokenizer_path: Optional[str] = None):
        """
        Initialize the summarization model
        
        Args:
            model_path: Path to fine-tuned model (if None, uses fine-tuned if available, else base model)
            tokenizer_path: Path to tokenizer (if None, uses fine-tuned if available, else base tokenizer)
        """
        self.device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
          
          
        if model_path is None:
            self.model_name = FINE_TUNED_MODEL_PATH if os.path.exists(FINE_TUNED_MODEL_PATH) else MODEL_NAME
        else:
            self.model_name = model_path if os.path.exists(model_path) else MODEL_NAME
            
        if tokenizer_path is None:
            self.tokenizer_name = TOKENIZER_PATH if os.path.exists(TOKENIZER_PATH) else MODEL_NAME
        else:
            self.tokenizer_name = tokenizer_path if os.path.exists(tokenizer_path) else MODEL_NAME
        
          
        self._load_model()
        
    def _load_model(self):
        """Load the model and tokenizer"""
        try:
            logger.info(f"Loading tokenizer from: {self.tokenizer_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
            
            logger.info(f"Loading model from: {self.model_name}")
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name).to(self.device)
            
              
            self.pipeline = pipeline(
                "summarization",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.device.type == "cuda" else -1
            )
            
            logger.info("Model and tokenizer loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def summarize_text(
        self, 
        text: str, 
        summary_length: str = "medium",
        **kwargs
    ) -> Dict[str, Union[str, int]]:
        """
        Summarize input text
        
        Args:
            text: Input text to summarize
            summary_length: "short", "medium", or "long"
            **kwargs: Additional generation parameters
            
        Returns:
            Dictionary with summary and metadata
        """
        try:
              
            length_config = SUMMARY_LENGTHS.get(summary_length, SUMMARY_LENGTHS["medium"])
            
              
            input_word_count = len(text.split())
            if input_word_count > 300 and summary_length == "short":
                logger.warning("Input too long for 'short' summary. Consider 'medium' or 'long'.")
                logger.info(f"Input has {input_word_count} words - 'short' may be too brief for good coverage.")
            
              
            gen_kwargs = {
                "length_penalty": LENGTH_PENALTY,
                "num_beams": NUM_BEAMS,
                "max_new_tokens": length_config["max_tokens"],
                **kwargs
            }
            
              
            result = self.pipeline(text, **gen_kwargs)
            summary = result[0]["summary_text"]
            
              
            original_length = len(text.split())
            summary_length_words = len(summary.split())
            compression_ratio = round(original_length / summary_length_words, 2) if summary_length_words > 0 else 0
            
            return {
                "summary": summary,
                "original_length": original_length,
                "summary_length": summary_length_words,
                "compression_ratio": compression_ratio,
                "summary_type": summary_length
            }
            
        except Exception as e:
            logger.error(f"Error during summarization: {str(e)}")
            raise
    
    def summarize_batch(
        self, 
        texts: List[str], 
        summary_length: str = "medium",
        **kwargs
    ) -> List[Dict[str, Union[str, int]]]:
        """
        Summarize multiple texts
        
        Args:
            texts: List of input texts
            summary_length: "short", "medium", or "long"
            **kwargs: Additional generation parameters
            
        Returns:
            List of dictionaries with summaries and metadata
        """
        return [self.summarize_text(text, summary_length, **kwargs) for text in texts]
    
    def detect_input_type(self, text: str) -> str:
        """
        Detect if input is chat-style or paragraph-style
        
        Args:
            text: Input text to analyze
            
        Returns:
            "chat" or "paragraph"
        """
          
        lines = text.strip().split('\n')
        
          
        chat_indicators = 0
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
              
            if ':' in line and len(line.split(':')[0]) < 20:
                chat_indicators += 1
            elif line.startswith('- ') or line.startswith('* '):
                chat_indicators += 1
                
          
        chat_ratio = chat_indicators / len([l for l in lines if l.strip()])
        
        return "chat" if chat_ratio > 0.3 else "paragraph"

def load_model(model_path: Optional[str] = None, tokenizer_path: Optional[str] = None) -> SummarizationModel:
    """
    Factory function to load the summarization model
    
    Args:
        model_path: Path to fine-tuned model
        tokenizer_path: Path to tokenizer
        
    Returns:
        SummarizationModel instance
    """
    return SummarizationModel(model_path, tokenizer_path)
