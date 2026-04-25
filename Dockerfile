FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir -e .
# CUDA 12.1 PyTorch wheel — requires the HF Space to be on T4 small or larger.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu121 torch
RUN pip install --no-cache-dir transformers peft accelerate huggingface_hub
EXPOSE 7860
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
