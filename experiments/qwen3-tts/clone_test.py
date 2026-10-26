"""Zero-shot Qwen3-TTS clone of a narrator from one reference clip: the same
sentences the narrator really read (held out), the voice creator's sample
passage, and a longer stretch to time it. Reports speed, peak video memory and
broken output (NaNs or silence) for each model and precision.

    python clone_test.py DATA_DIR OUT_DIR MODEL:DTYPE [MODEL:DTYPE ...]
    e.g. Qwen/Qwen3-TTS-12Hz-0.6B-Base:float32 Qwen/Qwen3-TTS-12Hz-1.7B-Base:float16
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from qwen_tts import Qwen3TTSModel

SAMPLE_TEXT = ("It was the kind of evening that made the whole town slow down. The lights came on one by one, "
               "and somewhere down the street a screen door opened, and someone called a name.")

data, out = Path(sys.argv[1]), Path(sys.argv[2])
ref_audio, ref_text = str(data / "ref.wav"), (data / "ref.txt").read_text(encoding="utf-8").strip()
held = [json.loads(line) for line in (data / "holdout.jsonl").read_text(encoding="utf-8").splitlines()][:8]
# Keep a share of video memory free: running out once took the whole machine down.
torch.cuda.set_per_process_memory_fraction(0.85, 0)

report = []
for spec in sys.argv[3:]:
    model_id, dtype_name = spec.rsplit(":", 1)
    tag = f"{model_id.split('/')[-1]}-{dtype_name}"
    folder = out / tag
    folder.mkdir(parents=True, exist_ok=True)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    try:
        model = Qwen3TTSModel.from_pretrained(model_id, device_map="cuda:0", dtype=getattr(torch, dtype_name),
                                              attn_implementation="sdpa")
        prompt = model.create_voice_clone_prompt(ref_audio=ref_audio, ref_text=ref_text, x_vector_only_mode=False)
    except Exception as exc:
        report.append({"model": tag, "error": f"loading: {exc}"})
        print(tag, "failed to load:", exc, flush=True)
        continue
    load_s = time.time() - t0
    jobs = [("sample", SAMPLE_TEXT)] + [(Path(h["audio"]).stem, h["text"]) for h in held]
    jobs.append(("long", " ".join(h["text"] for h in held)))  # one long stretch, to time it
    made, audio_s, synth_s, broken = 0, 0.0, 0.0, []
    for name, text in jobs:
        t1 = time.time()
        try:
            wavs, sr = model.generate_voice_clone(text=text, language="English", voice_clone_prompt=prompt)
        except Exception as exc:
            broken.append(f"{name}: {exc}")
            print(tag, name, "failed:", exc, flush=True)
            continue
        synth_s += time.time() - t1
        wav = np.asarray(wavs[0], dtype=np.float32)
        if not np.isfinite(wav).all() or np.abs(wav).max() < 1e-3:
            broken.append(f"{name}: NaN or silent")
            continue
        sf.write(folder / f"{name}.wav", wav, sr)
        made += 1
        audio_s += len(wav) / sr
        print(f"{tag} {name}: {len(wav) / sr:.1f} s of audio in {time.time() - t1:.1f} s", flush=True)
    report.append({"model": tag, "load_s": round(load_s, 1), "made": made, "of": len(jobs),
                   "audio_s": round(audio_s, 1), "synth_s": round(synth_s, 1),
                   "realtime_x": round(audio_s / synth_s, 2) if synth_s else None,
                   "peak_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 2), "broken": broken})
    del model
    torch.cuda.empty_cache()

(out / "report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
print(json.dumps(report, indent=1))
