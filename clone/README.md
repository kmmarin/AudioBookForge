# Clone server

Voices cloned from a reference clip of a narrator, with
[Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) (Apache-2.0). Forge narrates
books with them, and the voice creator makes samples with them. Both talk to
this server over HTTP (`app/clone.py`), so it can run on whichever machine has
the GPU for it.

It's an image of its own because qwen-tts needs transformers 4.57.3, and
Forge's image has 5.x for Kokoro, which qwen-tts's model code doesn't run on.

## Running it

One line on the GPU machine sets it up. It ends with the address and token to
paste into Forge's **Settings → GPU servers**; running it again updates it and
keeps the token.

| Where | One line |
|---|---|
| Windows, natively (NVIDIA or AMD Radeon), PowerShell as administrator | `irm https://raw.githubusercontent.com/kmmarin/AudioBookForge/main/deploy/gpu-server.ps1 \| iex` |
| Windows with WSL and Docker (NVIDIA), PowerShell as administrator | `& ([scriptblock]::Create((irm https://raw.githubusercontent.com/kmmarin/AudioBookForge/main/deploy/gpu-server.ps1))) -Mode wsl` |
| Linux with Docker (NVIDIA) | `curl -fsSL https://raw.githubusercontent.com/kmmarin/AudioBookForge/main/deploy/gpu-server.sh \| sudo bash` |

By hand, with Docker and the NVIDIA runtime: put `CLONE_TOKEN=<token>` in `.env`
and run `docker compose -f docker-compose.clone.yml up -d --build`. Then add it
in Forge's Settings, or set `CLONE_URL=http://<that machine>:8100` and the same
`CLONE_TOKEN` in Forge's `.env`.

The first use of each model downloads it from Hugging Face. That's about 2 GB
for 0.6B, 4.5 GB for 1.7B and 7 GB for the picture model.

### On Windows itself (no WSL, no Docker)

`deploy\gpu-server.ps1` (the first one-liner) runs `clone\windows\setup.ps1` for you. It:
1. installs Python (with winget, if it's missing), PyTorch for the card and
   qwen-tts in one folder (default `C:\forge-clone`);
2. saves the token (`save-token.ps1`);
3. registers a task that starts it at boot as SYSTEM (`install-autostart.ps1`).

The card decides the PyTorch build:
- **NVIDIA:** PyTorch's CUDA 12.8 build.
- **AMD Radeon:** AMD's nightly ROCm build for its family (RX 5000 to 9000).
  ROCm in WSL doesn't support older Radeons such as the RX 6000 series, but these
  builds do.

`start.ps1` and `stop.ps1` in `C:\forge-clone\server` start and stop it by hand.

On an RX 6800, Qwen3-TTS 1.7B runs at about 3.8x real time in batches of 16, in float16.

- **MIOpen is off:** it searched for convolution kernels for every sentence length, and that made it 7x slower.
- **Never end the server by force while it's working:** that once left the driver stuck until a restart. `stop.ps1` lets go of the model first.

### On a Windows PC with an NVIDIA GPU (WSL)

`deploy\gpu-server.ps1 -Mode wsl` (the second one-liner) does this:
1. It runs `deploy/clone-setup.sh` in WSL (Ubuntu with systemd). That installs
   Docker and NVIDIA's container toolkit if they're missing, copies `clone/` and
   `docker-compose.clone.yml` to `/opt/audiobook-forge-clone`, and starts it.
   The token comes in on stdin, so it never shows up in a command line.
2. WSL's ports only reach other machines through Windows, so it then adds a port
   proxy from the PC's address and a firewall rule for port 8100.

## Models and memory

| Model | bf16 (RTX 30xx and newer) | float32 (older cards) |
|---|---|---|
| `qwen3-tts-1.7b`, the best | about 7 GB | about 10 GB |
| `qwen3-tts-0.6b` | about 3 GB | about 5.5 GB |

`/api/health` says which ones fit this GPU.

- **GTX 16xx cards** run in float32, because they give NaNs in float16. On a 6 GB GTX 1660, 0.6B peaked at about 4.9 GB and was about 6.5x slower than real time.
- **The CPU** works but is unusably slow for books: about 0.02x real time.
- **Memory:** one model is loaded at a time, on first use. It's let go after `CLONE_IDLE_SECONDS` (default 300) without work, or on `POST /api/unload`.

## Settings

| Variable | Default | What it does |
|---|---|---|
| `CLONE_TOKEN` | none | Required in `X-Forge-Token` on everything but `/api/health` |
| `CLONE_IDLE_SECONDS` | 300 | When to let go of an unused model |
| `CLONE_BATCH` | by video memory: 16 from 14 GB, 8 from 10 GB, else 4 | Sentences generated at once; falls back to one at a time if it runs out of video memory |
| `CLONE_MIOPEN` | off | AMD cards: `on` uses MIOpen for convolutions. Off by default; it searched for kernels for every sentence length and took 95% of the time |
| `CLONE_GPU_SHARE` | 0.9 | The most video memory it may take, so running out is an error, not a crash |
| `CLONE_DTYPE` | auto | `bfloat16`, `float16` or `float32` |
| `CLONE_PORT` | 8100 | |

## Pictures (Watch and read)

The same server draws Watch and read's pictures: `POST /api/draw`, with DreamShaper XL Lightning (SDXL, 4 steps, about 7 GB, downloaded on first use).
- **One model at a time:** drawing shares the GPU's one model slot with the voices.
- **Voices come first:** while a cloned voice is being spoken (or was in the last minute), drawing answers 503 and Forge asks again later.
- **On AMD cards:** MIOpen is switched on for drawing (`CLONE_DRAW_MIOPEN`, default on). Pictures are all one size, so its kernel search happens only once.

The drawing settings are adapted from [SeeStory](https://github.com/pgotta/SeeStory) (MIT).

## Several servers

Forge's **Settings** page lists the clone servers it can use and what each one does: voices, pictures or both. A job goes to the first server that does it and is answering. `CLONE_URL` in `.env` adds one more to that list.

## API

The API is described in `app/clone.py`. Reference clips go to `PUT /api/refs/<key>`, where the key is a sha1 of the clip plus its text. `POST /api/speak` returns one WAV per sentence. `POST /api/draw` returns one JPEG picture.

`clone/test_server.py` checks a running server end to end:

```
PYTHONPATH=. python clone/test_server.py URL REF.wav REF.txt [TOKEN] [OUT_DIR]
```
