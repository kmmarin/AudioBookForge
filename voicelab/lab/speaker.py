"""Speaker embeddings (SpeechBrain's ECAPA-TDNN, trained on VoxCeleb):
a fingerprint of a voice, used to keep only the narrator's own clips and to
measure how much a new voice sounds like them."""
from __future__ import annotations

import numpy as np

from . import config

RATE = 16000
EXCERPT = 4.0  # seconds: enough to tell voices apart, and half the work of a whole clip
_model = None


def middle(audio: np.ndarray, rate: int, seconds: float = EXCERPT) -> np.ndarray:
    """The middle few seconds of a clip, where the voice is steadiest."""
    n = int(rate * seconds)
    if len(audio) <= n:
        return audio
    start = (len(audio) - n) // 2
    return audio[start:start + n]


def device() -> str:
    """The speaker model runs where Whisper does (LAB_WHISPER_DEVICE)."""
    return "cuda" if config.WHISPER_DEVICE == "cuda" else "cpu"


def _get():
    global _model
    if _model is None:
        import torch
        from huggingface_hub import snapshot_download
        from speechbrain.inference.speaker import EncoderClassifier
        torch.set_num_threads(config.THREADS)
        # Fetched here, then loaded as a local folder: SpeechBrain's own
        # download code passes an argument newer huggingface_hub versions refuse.
        local = snapshot_download(config.SPEAKER_MODEL, allow_patterns=[
            "hyperparams.yaml", "embedding_model.ckpt", "mean_var_norm_emb.ckpt", "classifier.ckpt",
            "label_encoder.txt"])
        _model = EncoderClassifier.from_hparams(source=local, savedir=str(config.LAB_MODELS / "ecapa"),
                                                overrides={"pretrained_path": local},
                                                run_opts={"device": device()})
    return _model


def embed(clips: list[np.ndarray], batch: int = 16) -> np.ndarray:
    """One unit-length 192-number fingerprint per clip (16 kHz mono)."""
    import torch
    if not clips:
        return np.zeros((0, 192), np.float32)
    model = _get()
    out = np.zeros((len(clips), 192), np.float32)
    order = sorted(range(len(clips)), key=lambda i: len(clips[i]))  # similar lengths: little padding
    with torch.inference_mode():
        for n in range(0, len(order), batch):
            idx = order[n:n + batch]
            longest = max(len(clips[i]) for i in idx)
            wavs = torch.zeros(len(idx), longest)
            lens = torch.zeros(len(idx))
            for row, i in enumerate(idx):
                wavs[row, :len(clips[i])] = torch.from_numpy(clips[i])
                lens[row] = len(clips[i]) / longest
            emb = model.encode_batch(wavs.to(device()), lens.to(device())).squeeze(1).cpu().numpy()
            emb /= np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-9)
            out[idx] = emb
    return out


def to_blob(vec: np.ndarray) -> bytes:
    return np.asarray(vec, np.float32).tobytes()


def from_blob(blob: bytes | None) -> np.ndarray | None:
    return None if not blob else np.frombuffer(blob, np.float32)


def centroid(embs: np.ndarray, keep: float = 0.8, rounds: int = 3) -> np.ndarray:
    """The main voice: the mean fingerprint, refined a few times on the clips
    closest to it, so other voices and odd clips don't pull it away."""
    center = embs.mean(axis=0)
    for _ in range(rounds):
        center /= max(np.linalg.norm(center), 1e-9)
        sims = embs @ center
        cut = np.quantile(sims, 1 - keep)
        center = embs[sims >= cut].mean(axis=0)
    return center / max(np.linalg.norm(center), 1e-9)
