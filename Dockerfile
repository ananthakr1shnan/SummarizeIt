FROM python:3.11-slim

WORKDIR /app

 
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV TRANSFORMERS_CACHE=/app/models/.cache
ENV HF_HOME=/app/models/.cache

 
RUN apt-get update && apt-get install -y \
    wget \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

 
COPY requirements_spaces.txt requirements.txt

 
RUN pip install --no-cache-dir -r requirements.txt

 
RUN pip install fastapi uvicorn[standard] jinja2 python-multipart

 
COPY . .

 
RUN mkdir -p models data logs src/app/static src/app/templates && \
    chmod -R 755 /app

 
RUN python -c "import nltk; nltk.download('punkt')"

 
RUN echo "Checking model files..." && \
    ls -la models/ && \
    echo "Model check complete"

 
EXPOSE 7860

 
HEALTHCHECK --interval=30s --timeout=30s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:7860/ || exit 1

 
CMD ["python", "-m", "uvicorn", "src.app.main:app", "--host", "0.0.0.0", "--port", "7860"]
