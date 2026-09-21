"""Wrapper around Banglabox/chatterbox-bangla-tts with long-text chunking."""
from __future__ import annotations

import base64
import io
import os
import re
import subprocess
import sys
import tempfile
import urllib.request

import numpy as np
import soundfile as sf

TTS_DIR = os.getenv("TTS_DIR", "/models/chatterbox-bangla-tts")
sys.path.insert(0, os.path.join(TTS_DIR, "inference"))
from infer import BanglaTTS  # noqa: E402  (provided by the model repository)

BUILTIN_VOICES = {
    "default": os.path.join(TTS_DIR, "voices", "reference_merged.wav"),
    "female": os.path.join(TTS_DIR, "voices", "ref_female.wav"),
}
MAX_CHARS_PER_CHUNK = int(os.getenv("TTS_MAX_CHARS", "220"))
PAUSE_SECONDS = 0.25
SENTENCE_END = re.compile(r"(?<=[।!?\.\n])\s+")


class TtsEngine:
    def __init__(self) -> None:
        self.model = BanglaTTS(TTS_DIR)

    def resolve_voice(self, voice: str | None, voice_url: str | None, voice_base64: str | None) -> tuple[str, bool]:
        """Returns (path to reference wav, is_temporary)."""
        if voice_base64 or voice_url:
            data = base64.b64decode(voice_base64) if voice_base64 else urllib.request.urlopen(voice_url, timeout=30).read()
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            tmp.write(data)
            tmp.close()
            return tmp.name, True
        path = BUILTIN_VOICES.get(voice or "default")
        if not path or not os.path.exists(path):
            raise ValueError(f"unknown voice '{voice}'. Built-in voices: {list(BUILTIN_VOICES)}")
        return path, False

    @staticmethod
    def split_text(text: str) -> list[str]:
        sentences = [s.strip() for s in SENTENCE_END.split(text.strip()) if s.strip()]
        chunks: list[str] = []
        current = ""
        for s in sentences:
            if len(current) + len(s) + 1 <= MAX_CHARS_PER_CHUNK:
                current = f"{current} {s}".strip()
            else:
                if current:
                    chunks.append(current)
                # A single very long sentence is split on commas/spaces.
                while len(s) > MAX_CHARS_PER_CHUNK:
                    cut = max(s.rfind(",", 0, MAX_CHARS_PER_CHUNK), s.rfind(" ", 0, MAX_CHARS_PER_CHUNK))
                    cut = cut if cut > 0 else MAX_CHARS_PER_CHUNK
                    chunks.append(s[:cut].strip())
                    s = s[cut:].strip(" ,")
                current = s
        if current:
            chunks.append(current)
        return chunks

    def synthesize(self, text: str, ref_path: str) -> tuple[np.ndarray, int]:
        pieces, sr = [], None
        for chunk in self.split_text(text):
            wav, sr = self.model.tts(chunk, ref_path)
            wav = np.asarray(wav.detach().cpu().numpy() if hasattr(wav, "detach") else wav, dtype=np.float32).squeeze()
            pieces.append(wav)
            pieces.append(np.zeros(int(sr * PAUSE_SECONDS), dtype=np.float32))
        if not pieces:
            raise ValueError("text is empty")
        return np.concatenate(pieces[:-1]), int(sr)

    @staticmethod
    def encode(wav: np.ndarray, sr: int, fmt: str) -> bytes:
        buf = io.BytesIO()
        sf.write(buf, wav, sr, format="WAV")
        if fmt == "wav":
            return buf.getvalue()
        # mp3 via ffmpeg (smaller files for mobile users)
        proc = subprocess.run(
            ["ffmpeg", "-loglevel", "error", "-f", "wav", "-i", "pipe:0", "-b:a", "128k", "-f", "mp3", "pipe:1"],
            input=buf.getvalue(),
            capture_output=True,
            check=True,
        )
        return proc.stdout
