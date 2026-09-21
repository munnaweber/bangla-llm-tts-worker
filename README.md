# Bangla LLM + TTS worker (RunPod Serverless)

One GPU worker with two models:

| Model | Licence | Job |
|---|---|---|
| `Qwen/Qwen3-4B-Instruct-2507` (default, change with `LLM_MODEL`) | Apache 2.0 | Bangla → English image prompts, translation, Bangla captions |
| `Banglabox/chatterbox-bangla-tts` (baked into the image) | MIT | Bangla text → natural voice (mp3/wav), voice cloning from a reference clip |

## Tasks

| `task` | Input | Output |
|---|---|---|
| `image_prompt` | `text` (Bangla idea) | `image_prompt` (English), `overlay_texts` (Bangla words for the image) |
| `translate` | `text`, `target` (default English) | `translation` |
| `caption` | `text`, `language`, `count`, `tone` | `captions[]` |
| `tts` | `text`, `voice` (`default` / `female`) or `voice_url` / `voice_base64`, `format` (`mp3`/`wav`) | `audio_base64`, `duration_seconds` |

Long TTS text is split into sentences (max 220 chars each) and joined with short pauses. Default max 3000 characters per request.

## Deploy

1. Push this folder to GitHub.
2. RunPod → **Serverless → New Endpoint → Deploy from a GitHub repository**.
3. Settings:

| Setting | Value |
|---|---|
| GPU | **24 GB** (RTX 4090 / L4 / A5000) — both models fit (~12–14 GB) |
| Container disk | 40 GB |
| Network volume | Optional (LLM cache at `/runpod-volume/huggingface`); TTS is already inside the image |
| Active workers | 0 |
| Max workers | 2–3 |
| Idle timeout | 30–60 s |
| Execution timeout | 300 s |
| FlashBoot | On |

4. Environment variables (optional): `LLM_MODEL`, `LLM_4BIT` (`true` for 16 GB GPUs), `HF_TOKEN` (for gated models like `google/gemma-3-4b-it`), `MAX_TTS_CHARS`.

## Test

```bash
export RUNPOD_API_KEY=...  RUNPOD_ENDPOINT_ID=...
python test_client.py tests/image_prompt.json
python test_client.py tests/tts.json        # saves voice.mp3
python test_client.py tests/caption.json
python test_client.py tests/translate.json
```

## Voices

Built-in: `default` (reference_merged.wav) and `female` (ref_female.wav) from the model repo.
Custom voice: send `voice_url` or `voice_base64` with a clean 5–10 s WAV. Use only voices you have written permission to clone.
