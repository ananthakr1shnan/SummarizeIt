from fastapi import FastAPI, HTTPException, File, UploadFile, Form, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Optional, List
import logging
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.model.summarizer import SummarizationModel
from config.settings import FINE_TUNED_MODEL_PATH, TOKENIZER_PATH

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    author="Ananthakrishnan K",
    author_email="ananthakrishnan073@gmail.com",
    title="SummarizeIt API",
    description="Text and Chat Summarization API using Fine-tuned Pegasus",
    version="1.0.0"
)
templates = Jinja2Templates(directory="src/app/templates")
app.mount("/static", StaticFiles(directory="src/app/static"), name="static")

model = None

class SummarizeRequest(BaseModel):
    text: str
    summary_length: Optional[str] = "medium"
    input_type: Optional[str] = "auto"  

class SummarizeResponse(BaseModel):
    summary: str
    original_length: int
    summary_length: int
    compression_ratio: float
    summary_type: str
    input_type: str

class BatchSummarizeRequest(BaseModel):
    texts: List[str]
    summary_length: Optional[str] = "medium"
    input_type: Optional[str] = "auto"

@app.on_event("startup")
async def startup_event():
    global model
    try:
        logger.info("Loading summarization model...")
        
        if os.path.exists(FINE_TUNED_MODEL_PATH) and os.path.exists(TOKENIZER_PATH):
            model = SummarizationModel(FINE_TUNED_MODEL_PATH, TOKENIZER_PATH)
            logger.info("Fine-tuned model loaded successfully")
        else:
            model = SummarizationModel()
            logger.info("Base model loaded (fine-tuned model not found)")
            
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_type": "fine-tuned" if os.path.exists(FINE_TUNED_MODEL_PATH) else "base"
    }

@app.post("/summarize", response_model=SummarizeResponse)
async def summarize_text(request: SummarizeRequest):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        input_type = request.input_type
        if input_type == "auto":
            input_type = model.detect_input_type(request.text)
        
        result = model.summarize_text(
            text=request.text,
            summary_length=request.summary_length
        )
        result["input_type"] = input_type
        
        return SummarizeResponse(**result)
        
    except Exception as e:
        logger.error(f"Summarization error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Summarization failed: {str(e)}")

@app.post("/summarize/batch")
async def summarize_batch(request: BatchSummarizeRequest):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        results = []
        for text in request.texts:
            input_type = request.input_type
            if input_type == "auto":
                input_type = model.detect_input_type(text)
            
            result = model.summarize_text(
                text=text,
                summary_length=request.summary_length
            )
            result["input_type"] = input_type
            results.append(result)
        
        return {"summaries": results}
        
    except Exception as e:
        logger.error(f"Batch summarization error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch summarization failed: {str(e)}")

@app.post("/summarize/file")
async def summarize_file(
    file: UploadFile = File(...),
    summary_length: str = Form("medium"),
    input_type: str = Form("auto")
):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
         
        content = await file.read()
        text = content.decode('utf-8')
        
         
        if input_type == "auto":
            input_type = model.detect_input_type(text)
        
         
        result = model.summarize_text(
            text=text,
            summary_length=summary_length
        )
        result["input_type"] = input_type
        result["filename"] = file.filename
        
        return result
        
    except Exception as e:
        logger.error(f"File summarization error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"File summarization failed: {str(e)}")

@app.get("/api/summary-lengths")
async def get_summary_lengths():
    return {
        "options": [
            {"value": "short", "label": "Short (~25 tokens)", "description": "Concise highlights - best for key points"},
            {"value": "medium", "label": "Medium (~50 tokens)", "description": "Balanced summary - good for most use cases"},
            {"value": "long", "label": "Long (~80 tokens)", "description": "Comprehensive overview - detailed coverage"}
        ]
    }

 
@app.exception_handler(404)
async def not_found_handler(request: Request, exc):
    return templates.TemplateResponse("404.html", {"request": request}, status_code=404)

@app.exception_handler(500)
async def internal_error_handler(request: Request, exc):
    logger.error(f"Internal server error: {str(exc)}")
    return templates.TemplateResponse("500.html", {"request": request}, status_code=500)

 
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
