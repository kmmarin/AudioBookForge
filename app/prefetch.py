"""Download every model the voices file uses, so the first book starts at full speed.

    docker compose run --rm worker python -m app.prefetch
"""
from __future__ import annotations

from . import config, qa, tts, voices


def main() -> None:
    tts.init_worker(config.TORCH_THREADS, config.DEVICE)
    for voice in voices.list_voices():
        print(f"Fetching voice: {voice['label']}", flush=True)
        try:
            tts.get_engine(voice).synth("Hello.", voice, 1.0)
        except Exception as exc:
            print(f"  failed: {exc}", flush=True)
    if config.QA_DEFAULT:
        print(f"Fetching Whisper model: {config.WHISPER_MODEL}", flush=True)
        qa._get_model()
    if config.TRANSCRIBE_MODEL != config.WHISPER_MODEL or not config.QA_DEFAULT:
        print(f"Fetching Whisper model for added audiobooks: {config.TRANSCRIBE_MODEL}", flush=True)
        from faster_whisper.utils import download_model
        download_model(config.TRANSCRIBE_MODEL, cache_dir=str(config.MODELS_DIR / "whisper"))
    print("All models are cached.", flush=True)


if __name__ == "__main__":
    main()
