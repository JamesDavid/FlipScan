"""Generate a narration voice from a text description (Parler-TTS).

Chatterbox clones voices from reference audio but can't invent one from words.
Parler-TTS (Apache-2.0) can: it synthesizes speech whose speaker matches a
natural-language description ("an elderly male speaker with a deep, gravelly
voice, speaking slowly and formally"). We use it to mint a ~20 s generic voice
sample per character from the cast analysis's description, drop it in the
shared voice library, and let Chatterbox clone it for the actual narration.

Parler conditions on voice ATTRIBUTES (age, gender, pitch, pace, tone) — it
does not know or imitate real people, so a description mentioning a famous
name only contributes its adjectives.
"""

from __future__ import annotations

import gc
import re
from pathlib import Path

# spoken while minting the sample — attribute-rich, neutral content
_SAMPLE_TEXT = ("The evening settled over the airfield as the last light "
                "faded from the hangar doors. I have seen many machines in my "
                "time, but none that moved with such quiet certainty. "
                "Tomorrow we begin again, and the weather looks promising.")

_MODEL_ID = "parler-tts/parler-tts-mini-v1"


def build_description(character_desc: str, sounds_like: str) -> str:
    """Compose Parler's voice-description prompt from what the cast analysis
    knows. Famous names carry no meaning for Parler, so keep only the
    adjectives around them; always anchor with recording-quality cues."""
    bits = []
    for part in (character_desc, sounds_like):
        part = (part or "").strip().rstrip(".")
        if part:
            # "similar to X in Y" adds nothing for Parler — keep the qualities
            part = re.sub(r"\s*[—-]?\s*(similar to|in the vein of|like)\s+[A-Z][^,;.]*",
                          "", part).strip(" ,;—-")
            if part:
                bits.append(part)
    desc = "; ".join(bits) or "a neutral, pleasant adult voice"
    return (f"A speaker with this character: {desc}. Moderate pace, "
            f"expressive but controlled delivery, very clear audio, "
            f"close-up recording with no background noise.")


def generate_voice_sample(description: str, out_wav: Path,
                          log=print) -> Path:
    """Mint a voice sample matching `description` -> 24 kHz mono wav (the
    library format). The Parler model (~2.5 GB) loads for the call and is
    released afterwards so audiobook synthesis gets the GPU back."""
    import torch
    log(f"  voice-gen: loading {_MODEL_ID}…")
    from parler_tts import ParlerTTSForConditionalGeneration
    from transformers import AutoTokenizer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ParlerTTSForConditionalGeneration.from_pretrained(_MODEL_ID).to(device)
    tok = AutoTokenizer.from_pretrained(_MODEL_ID)
    try:
        log(f"  voice-gen: synthesizing sample — {description[:90]!r}")
        ids = tok(description, return_tensors="pt").input_ids.to(device)
        prompt = tok(_SAMPLE_TEXT, return_tensors="pt").input_ids.to(device)
        with torch.no_grad():
            audio = model.generate(input_ids=ids, prompt_input_ids=prompt)
        wav = audio.cpu().float().squeeze(0)
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        sr = int(model.config.sampling_rate)
        import torchaudio
        wav = torchaudio.functional.resample(wav, sr, 24000)
        out_wav.parent.mkdir(parents=True, exist_ok=True)
        torchaudio.save(str(out_wav), wav, 24000,
                        encoding="PCM_S", bits_per_sample=16)
        log(f"  voice-gen: saved {out_wav.name} "
            f"({wav.shape[-1] / 24000:.1f}s)")
        return out_wav
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
