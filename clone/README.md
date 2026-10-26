# Clone server

Voices cloned from a reference clip of a narrator, with
[Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) (Apache-2.0). Forge narrates
books with them, and the voice creator makes samples with them. Both talk to
this server over HTTP (`app/clone.py`), so it can run on whichever machine has
the GPU for it.

It's an image of its own because qwen-tts needs transformers 4.57.3, and
Forge's image has 5.x for Kokoro, which qwen-tts's model code doesn't run on.

## Running it

On the machine with the GPU (Docker with the NVIDIA runtime):

```
docker compose -f docker-compose.clone.yml up -d --build
```

Then, in Forge's `.env` on the machine running Forge and the voice creator, set:

- `CLONE_URL=http://<that machine>:8100`
- `CLONE_TOKEN`, the same value on both machines.

The first use of each model downloads it from Hugging Face into the `clone-hf`
volume. That's about 2 GB for 0.6B and 4.5 GB for 1.7B.

### On Windows with an AMD Radeon (no WSL, no Docker)

ROCm in WSL doesn't support older Radeons, such as the RX 6000 series, but AMD's
nightly ROCm builds of PyTorch for Windows do. `clone/windows/` sets the server up
on Windows itself:

1. `setup.ps1` puts a Python environment, PyTorch for the card's family and
   qwen-tts in one folder (default `D:\forge-clone`).
2. Pipe Forge's token into `save-token.ps1`.
3. `start.ps1` and `stop.ps1` start and stop the server.
4. `install-autostart.ps1` starts it at boot, as SYSTEM (admin PowerShell).

On an RX 6800, Qwen3-TTS 1.7B runs at about 3.8x real time in batches of 16, in float16.

- **MIOpen is off:** it searched for convolution kernels for every sentence length, and that made it 7x slower.
- **Never end the server by force while it's working:** that once left the driver stuck until a restart. `stop.ps1` lets go of the model first.

### On a Windows PC with an NVIDIA GPU (WSL)

`deploy/clone-setup.sh` installs Docker and NVIDIA's container toolkit in WSL
(Ubuntu with systemd) if they're missing, unpacks a bundle of `clone/` plus
`docker-compose.clone.yml` into `/opt/audiobook-forge-clone`, and starts it.
The token comes on stdin, so it never shows up in a command line.

WSL's ports only reach other machines through Windows. In an admin PowerShell
on that PC (with its own LAN address):

```
netsh interface portproxy add v4tov4 listenaddress=<its LAN IP> listenport=8100 connectaddress=127.0.0.1 connectport=8100
New-NetFirewallRule -DisplayName "Audiobook Forge clone server" -Direction Inbound -Protocol TCP -LocalPort 8100 -Action Allow
```

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
