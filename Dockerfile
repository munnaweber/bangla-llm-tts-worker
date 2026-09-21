# RunPod serverless worker: Translator LLM + Bangla TTS (Chatterbox-Bangla) on ONE GPU.
FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    TTS_DIR=/models/chatterbox-bangla-tts

RUN apt-get update && apt-get install -y --no-install-recommends git ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Bake the Bangla TTS model (MIT) into the image -> no download on cold start.
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('Banglabox/chatterbox-bangla-tts', local_dir='/models/chatterbox-bangla-tts')" \
    && pip install --no-cache-dir -r /models/chatterbox-bangla-tts/inference/requirements.txt

COPY handler.py prompts.py tts_engine.py ./

CMD ["python", "-u", "handler.py"]
