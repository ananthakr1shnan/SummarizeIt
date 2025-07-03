"""
Hugging Face Spaces Gradio App for SummarizeIt
Text and Chat Summarization using Fine-tuned Pegasus
"""
import gradio as gr
import os
import sys
import logging
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent))

# Import your model
from src.model.summarizer import SummarizationModel

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model instance
model = None

def load_model():
    """Load the summarization model"""
    global model
    try:
        logger.info("Loading summarization model...")
        
        # Check if fine-tuned model exists
        fine_tuned_path = "./models/pegasus-samsum-model"
        tokenizer_path = "./models/tokenizer"
        
        if os.path.exists(fine_tuned_path) and os.path.exists(tokenizer_path):
            model = SummarizationModel(fine_tuned_path, tokenizer_path)
            logger.info("✅ Fine-tuned model loaded successfully!")
        else:
            model = SummarizationModel()
            logger.info("⚠️ Fine-tuned model not found, using base model")
            
        return "✅ Model loaded successfully!"
        
    except Exception as e:
        error_msg = f"❌ Failed to load model: {str(e)}"
        logger.error(error_msg)
        return error_msg

def summarize_text(text, summary_length="medium", input_type="auto"):
    """
    Summarize the input text
    """
    if model is None:
        return "❌ Model not loaded. Please refresh the page."
    
    if not text or not text.strip():
        return "⚠️ Please enter some text to summarize."
    
    try:
        # Detect input type if auto
        if input_type == "auto":
            input_type = model.detect_input_type(text)
        
        # Generate summary
        result = model.summarize_text(text, summary_length)
        
        # Format output
        summary = result['summary']
        stats = f"""
**Summary Statistics:**
- Original: {result['original_length']} words
- Summary: {result['summary_length']} words  
- Compression: {result['compression_ratio']}x
- Input type: {input_type}
- Summary type: {result['summary_type']}
"""
        
        return summary, stats
        
    except Exception as e:
        error_msg = f"❌ Summarization failed: {str(e)}"
        logger.error(error_msg)
        return error_msg, ""

def create_interface():
    """Create the Gradio interface"""
    
    # Custom CSS for better styling
    css = """
    .gradio-container {
        font-family: 'IBM Plex Sans', sans-serif;
    }
    .output-text {
        font-size: 16px;
        line-height: 1.5;
    }
    """
    
    with gr.Blocks(css=css, title="SummarizeIt - AI Text Summarizer") as interface:
        
        gr.Markdown("""
        # 📝 SummarizeIt - AI Text Summarizer
        
        **Powered by Fine-tuned Pegasus Model**
        
        Transform long texts and chat conversations into concise, meaningful summaries. 
        Perfect for articles, emails, meeting notes, and chat conversations.
        """)
        
        with gr.Row():
            with gr.Column(scale=2):
                # Input section
                gr.Markdown("### 📥 Input Text")
                text_input = gr.Textbox(
                    label="Text to Summarize",
                    placeholder="Paste your text here... (articles, emails, chat conversations, etc.)",
                    lines=10,
                    max_lines=20
                )
                
                with gr.Row():
                    summary_length = gr.Radio(
                        choices=["short", "medium", "long"],
                        value="medium",
                        label="Summary Length",
                        info="Short: ~25 tokens, Medium: ~50 tokens, Long: ~80 tokens"
                    )
                    
                    input_type = gr.Radio(
                        choices=["auto", "chat", "paragraph"],
                        value="auto",
                        label="Input Type",
                        info="Auto-detect or specify the type of content"
                    )
                
                summarize_btn = gr.Button("✨ Summarize", variant="primary", size="lg")
                
            with gr.Column(scale=2):
                # Output section
                gr.Markdown("### 📤 Summary")
                summary_output = gr.Textbox(
                    label="Generated Summary",
                    lines=8,
                    max_lines=15,
                    interactive=False,
                    elem_classes=["output-text"]
                )
                
                stats_output = gr.Markdown(
                    label="Statistics",
                    value="*Summary statistics will appear here*"
                )
        
        # Example section
        gr.Markdown("### 📚 Try These Examples")
        
        examples = [
            [
                """John: Hey Sarah, how was your day at work?
Sarah: Pretty good! Had a big presentation today.
John: How did it go?
Sarah: Really well actually. The client loved our proposal.
John: That's awesome! We should celebrate.
Sarah: Definitely! Dinner tomorrow?
John: Sounds perfect. I'll make reservations.""",
                "medium",
                "chat"
            ],
            [
                """Artificial intelligence has made significant strides in recent years, particularly in the field of natural language processing. Large language models like GPT and BERT have revolutionized how we approach text understanding and generation. These models can perform a wide variety of tasks including translation, summarization, question answering, and creative writing. The training process involves feeding massive amounts of text data to neural networks, allowing them to learn patterns in human language. As these technologies continue to evolve, they promise to transform industries from healthcare to education, making information more accessible and communication more efficient.""",
                "long",
                "paragraph"
            ]
        ]
        
        gr.Examples(
            examples=examples,
            inputs=[text_input, summary_length, input_type],
            outputs=[summary_output, stats_output],
            fn=summarize_text,
            cache_examples=False
        )
        
        # Event handlers
        summarize_btn.click(
            fn=summarize_text,
            inputs=[text_input, summary_length, input_type],
            outputs=[summary_output, stats_output]
        )
        
        # Footer
        gr.Markdown("""
        ---
        ### 🔧 About
        This app uses a **fine-tuned Pegasus model** trained on the SAMSum dataset for high-quality summarization.
        Perfect for condensing long texts while preserving key information.
        
        ### 💡 Tips
        - **Chat conversations**: Use names/timestamps for better results
        - **Long articles**: Consider "long" summary for comprehensive coverage  
        - **Short texts**: "Medium" usually provides the best balance
        """)
    
    return interface

def main():
    """Main function to run the app"""
    # Load model on startup
    load_status = load_model()
    logger.info(f"Model loading status: {load_status}")
    
    # Create and launch interface
    interface = create_interface()
    
    # Launch with appropriate settings for HF Spaces
    interface.launch(
        share=False,  # Don't create public URL (HF Spaces handles this)
        server_name="0.0.0.0",  # Listen on all interfaces
        server_port=7860,  # Default HF Spaces port
        show_error=True,
        show_tips=True
    )

if __name__ == "__main__":
    main()
