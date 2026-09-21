# RunPod serverless worker: Translator LLM + Bangla TTS (Chatterbox-Bangla) on ONE GPU.
# CUDA 12.4 runs on far more RunPod hosts than 12.8 (which needs a very new NVIDIA driver), and
# matches the torch==2.6.0 pin of the Bangla TTS model.
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

# The Bangla TTS model pins torch==2.6.0 and transformers==4.46.3; the latter cannot load Qwen3, so TTS is
# off by default (LLM-only image). Build with --build-arg WITH_TTS=true for a TTS image (use a
# transformers-4.46-compatible LLM_MODEL there, or run TTS as its own worker).
ARG WITH_TTS=false

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    TTS_DIR=/models/chatterbox-bangla-tts \
    ENABLE_TTS=${WITH_TTS}

RUN apt-get update && apt-get install -y --no-install-recommends git ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Optional: bake the Bangla TTS model (MIT) into the image -> no download on cold start.
RUN if [ "$WITH_TTS" = "true" ]; then \
      python -c "from huggingface_hub import snapshot_download; snapshot_download('Banglabox/chatterbox-bangla-tts', local_dir='/models/chatterbox-bangla-tts')" \
      && pip install --no-cache-dir -r /models/chatterbox-bangla-tts/inference/requirements.txt; \
    fi

COPY handler.py prompts.py tts_engine.py ./

CMD ["python", "-u", "handler.py"]
