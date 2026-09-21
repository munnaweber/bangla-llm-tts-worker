"""
RunPod serverless handler: Translator LLM + Bangla TTS on one GPU.

Tasks:
  {"input": {"task": "image_prompt", "text": "<bangla idea>"}}
  {"input": {"task": "translate", "text": "...", "target": "English"}}
  {"input": {"task": "caption", "text": "<brief>", "language": "Bangla", "count": 3, "tone": "friendly"}}
  {"input": {"task": "tts", "text": "<bangla text>", "voice": "female", "format": "mp3"}}
"""
from __future__ import annotations

import base64
import json
import os
import re
import time

import runpod
import torch

from prompts import CAPTION_SYSTEM, IMAGE_PROMPT_SYSTEM, TRANSLATE_SYSTEM

LLM_MODEL = os.getenv("LLM_MODEL", "Qwen/Qwen3-4B-Instruct-2507")
LLM_4BIT = os.getenv("LLM_4BIT", "false").lower() == "true"
MAX_TTS_CHARS = int(os.getenv("MAX_TTS_CHARS", "3000"))
DEVICE = "cuda"
# TTS is opt-in: its model pins torch==2.6.0 / transformers==4.46.3, which cannot load Qwen3
# (needs transformers>=4.51). Enable it only with an image built with --build-arg WITH_TTS=true
# and a compatible LLM_MODEL, or run it as a separate worker.
ENABLE_TTS = os.getenv("ENABLE_TTS", "false").lower() == "true"

if os.path.isdir("/runpod-volume"):
    os.environ.setdefault("HF_HOME", "/runpod-volume/huggingface")


# ---------------------------------------------------------------- load models once per worker
def load_llm():
    from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer, BitsAndBytesConfig

    quant = (
        BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4")
        if LLM_4BIT
        else None
    )
    kwargs = dict(torch_dtype=torch.bfloat16, device_map=DEVICE, quantization_config=quant)
    if "gemma-3" in LLM_MODEL.lower():
        from transformers import Gemma3ForConditionalGeneration

        return Gemma3ForConditionalGeneration.from_pretrained(LLM_MODEL, **kwargs).eval(), AutoProcessor.from_pretrained(LLM_MODEL)
    return AutoModelForCausalLM.from_pretrained(LLM_MODEL, **kwargs).eval(), AutoTokenizer.from_pretrained(LLM_MODEL)


def check_gpu() -> None:
    """Fail fast with a readable message instead of a confusing CUDA error deep inside model loading."""
    print(f"[boot] torch {torch.__version__} (built for CUDA {torch.version.cuda})", flush=True)
    if not torch.cuda.is_available():
        raise RuntimeError(
            "No usable GPU: torch.cuda.is_available() is False. Usual causes: the RunPod host's NVIDIA "
            "driver is older than this image's CUDA build (set the endpoint's 'Allowed CUDA versions' to "
            f"{torch.version.cuda} or higher, or rebuild on an older CUDA base image), or the endpoint "
            "has no GPU type selected / the selected GPU type has no capacity."
        )
    props = torch.cuda.get_device_properties(0)
    print(f"[boot] GPU: {props.name}, {props.total_memory / 1e9:.1f} GB", flush=True)


t0 = time.time()
check_gpu()
print(f"[boot] loading LLM {LLM_MODEL} (4bit={LLM_4BIT})", flush=True)
LLM, TOKENIZER = load_llm()
TTS = None
if ENABLE_TTS:
    print("[boot] loading Bangla TTS", flush=True)
    from tts_engine import TtsEngine

    TTS = TtsEngine()
print(f"[boot] ready in {time.time() - t0:.1f}s, VRAM {torch.cuda.memory_allocated() / 1e9:.1f} GB", flush=True)


# ---------------------------------------------------------------- LLM helpers
def chat(system: str, user: str, max_new_tokens: int = 500, temperature: float = 0.0) -> str:
    if "gemma-3" in LLM_MODEL.lower():
        messages = [
            {"role": "system", "content": [{"type": "text", "text": system}]},
            {"role": "user", "content": [{"type": "text", "text": user}]},
        ]
    else:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    inputs = TOKENIZER.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
    ).to(DEVICE)
    gen = dict(max_new_tokens=max_new_tokens, do_sample=temperature > 0)
    if temperature > 0:
        gen.update(temperature=temperature, top_p=0.9)
    with torch.inference_mode():
        out = LLM.generate(**inputs, **gen)
    return TOKENIZER.decode(out[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True).strip()


def parse_json(text: str) -> dict:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    try:
        return json.loads(match.group(0)) if match else {}
    except json.JSONDecodeError:
        return {}


# ---------------------------------------------------------------- tasks
def task_image_prompt(inp: dict) -> dict:
    raw = chat(IMAGE_PROMPT_SYSTEM, inp["text"])
    data = parse_json(raw)
    return {
        "image_prompt": (data.get("image_prompt") or raw).strip(),
        "overlay_texts": [str(t) for t in data.get("overlay_texts") or []],
    }


def task_translate(inp: dict) -> dict:
    target = inp.get("target", "English")
    return {"translation": chat(TRANSLATE_SYSTEM.format(target=target), inp["text"])}


def task_caption(inp: dict) -> dict:
    system = CAPTION_SYSTEM.format(
        count=int(inp.get("count", 3)), language=inp.get("language", "Bangla"), tone=inp.get("tone", "friendly")
    )
    raw = chat(system, inp["text"], max_new_tokens=700, temperature=0.8)
    captions = parse_json(raw).get("captions") or [raw]
    return {"captions": captions}


def task_tts(inp: dict) -> dict:
    if TTS is None:
        raise ValueError("tts is disabled on this worker (needs an image built with WITH_TTS=true and ENABLE_TTS=true)")
    text = inp["text"].strip()
    if len(text) > MAX_TTS_CHARS:
        raise ValueError(f"text too long ({len(text)} chars, max {MAX_TTS_CHARS})")
    fmt = str(inp.get("format", "mp3")).lower()
    if fmt not in ("mp3", "wav"):
        raise ValueError("format must be mp3 or wav")
    ref_path, is_tmp = TTS.resolve_voice(inp.get("voice"), inp.get("voice_url"), inp.get("voice_base64"))
    try:
        wav, sr = TTS.synthesize(text, ref_path)
    finally:
        if is_tmp:
            os.unlink(ref_path)
    return {
        "audio_base64": base64.b64encode(TTS.encode(wav, sr, fmt)).decode(),
        "format": fmt,
        "sample_rate": sr,
        "duration_seconds": round(len(wav) / sr, 2),
    }


TASKS = {"image_prompt": task_image_prompt, "translate": task_translate, "caption": task_caption, "tts": task_tts}


def handler(job):
    inp = job.get("input") or {}
    task = inp.get("task", "image_prompt")
    if task not in TASKS:
        return {"error": f"unknown task '{task}'. Use one of {list(TASKS)}"}
    if not str(inp.get("text", "")).strip():
        return {"error": "text is required"}
    t = time.time()
    try:
        result = TASKS[task](inp)
    except (ValueError, KeyError) as e:
        return {"error": str(e)}
    result["task"] = task
    result["took_ms"] = int((time.time() - t) * 1000)
    return result


# Start the RunPod serverless worker (top-level so RunPod can detect it).
runpod.serverless.start({"handler": handler})
