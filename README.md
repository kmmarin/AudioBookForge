# Audiobook Forge

A self-hosted web app that turns ebooks into narrated, chaptered audiobooks,
lets you read along word by word on any device, and makes new narrator voices.
It can be one Docker host, or an always-on server for your library plus a
Windows PC with a GPU that does the heavy lifting.

- **Ebooks in, audiobooks out:** EPUB, PDF, Word and a dozen other formats in.
  M4B with chapters, tags and cover out, ready for Audiobookshelf.
- **Read along:** a player that highlights every word as it's spoken. Each
  listener keeps their own place across phone, tablet and browser. It also
  works for audiobooks you already have: Whisper writes down their text.
- **Your own voices (optional):** clone a narrator from a clip with
  Qwen3-TTS, or train a Piper voice that needs no GPU to read a book. You
  listen to samples and pick one by ear.

Nothing in this repository is a book, a recording or a voice. Every voice
model is downloaded at first use from the project that publishes it (see
[Voices and models](#voices-and-models)).

---

## Contents

- [Features](#features)
- [Quick start](#quick-start)
- [A server plus a Windows converter](#a-server-plus-a-windows-converter)
- [Making voices](#making-voices)
- [Cloned voices: the clone server](#cloned-voices-the-clone-server)
- [HTTPS](#https)
- [Configuration](#configuration)
- [How it works](#how-it-works)
- [Voices and models](#voices-and-models)
- [Development](#development)
- [License](#license)

---

## Features

### Turning books into audiobooks

| | |
|---|---|
| **Formats in** | EPUB, PDF (with a text layer), TXT (Project Gutenberg aware), DOCX, ODT, RTF, HTML, Markdown, FB2, reStructuredText, Org, LaTeX and more through pandoc, DOC through antiword. MOBI, AZW3 and other Kindle formats with `INSTALL_CALIBRE=true` (no DRM). |
| **Chapters** | Found from the table of contents, PDF bookmarks or headings. Front and back matter (copyright, contents, "also by") is turned off by default. Every chapter can be renamed, turned on or off, and its text edited. |
| **Book details** | Title, author, series, number, year, cover and more are read from the file. Missing details can be looked up online: Open Library, Audible, and Google Books with a free API key. Each result is scored against the file, and you pick when it's unclear. |
| **Voices** | 11 Kokoro voices and 6 Piper voices (US and UK English), one with several moods, plus voices you make. Voice, mood and speed are set per book or for many at once. **Compare voices** plays the same saved 10-second passage in each. |
| **Formats out** | M4B (the default), M4A, MP3, Opus or FLAC, with chapter markers, tags (narrator, series, genre and more) and the cover. Files go into an `Author/Series/1 - Title/Title.m4b` layout that Audiobookshelf reads, or a flat one. |
| **Speed** | Chapters are cut into small units that every worker voices in parallel, longest first, and several books share the workers. Units are cached under a hash of voice, speed and text, so a restart, a cancel or a text edit only re-voices what changed. Runs on the CPU, or on an NVIDIA GPU. |
| **Narration check** | Whisper transcribes a sample of every chapter and compares it with the text. A misread chapter is flagged, and **Redo** voices only the flagged chapters again. |
| **Versions** | Each voice a book has been made in is kept. Switching back to one takes seconds, and Read along changes voice on the same word. |

### Read along

- **The player:** every word is highlighted as it's spoken, and tapping the
  text jumps there. It has playback speed and chapter skipping, and works in
  any browser, with no app to install.
- **Listeners:** each person ("Who's listening?") keeps their own place and
  checkpoints in every book, and they follow them from device to device.
- **While listening:** a sleep timer (10 minutes to an hour, or the end of the
  chapter), checkpoints, and **Change the voice**, which offers the versions
  already made or a new voice.
- **Audiobooks you already have:** add an M4B, MP3 or other audiobook and
  choose **Make Read along**. Whisper writes down what's said, with the time
  of every word, a chapter at a time. Your file stays exactly as it was; it
  is what plays and downloads. A name Whisper misheard can be fixed in the
  text, and the timings follow without listening again.

### Watch and read

Simple hand-drawn pictures of what's happening, changing as the story is read.
The picture button in Read along's header offers two views:
- **Above the text:** the picture sits at the top, with the highlighted text
  below it.
- **Full page:** SeeStory's view. The picture fills the screen, and the
  sentence being read runs as a subtitle with its word lit.

How it works:
- **How scenes are chosen:** about every 75 words (30 seconds) a scene begins
  at a paragraph's start, and its picture is drawn from the passage's most
  paintable sentence.
- **Drawing:** the clone server's GPU draws the pictures, in pencil-and-ink
  style by default, ahead of time, and they're kept with the book.
- **Movement:** each picture is swept in as if being drawn, then drifts slowly,
  pausing with the audio.
- **Other voices:** the scenes are the same in every voice a book is made in;
  only their timing follows the voice.

Scene choice, prompts and movement are adapted from
[SeeStory](https://github.com/pgotta/SeeStory) by pgotta (MIT), which turns
an audiobook into an illustrated video. See
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

### Two machines, if you like

An always-on server (a Proxmox container, a NAS, any Docker host) keeps the
library and serves Read along. A Windows PC with a GPU voices books and sends
them over:
- A voice change asked for on the server is voiced on the PC, with progress
  shown on the server.
- If the PC is off, **Voice it here instead** does it on the server's CPU.

Setting this up: [A server plus a Windows converter](#a-server-plus-a-windows-converter).

### Your own voices (optional)

The voice creator, at `/createvoice`, makes narrator voices from recordings
of them, such as audiobooks they read.

1. **Upload the recordings.** Large files upload in parts and carry on after
   an interruption. A second copy of the same audio is skipped.
2. **Name the voice, or let it find the name** in the tags or in the
   "read by…" of the opening credits.
3. **It listens.** Whisper cuts clean clips at pauses, and drops music,
   noise, other speakers and strong character voices. Speaker fingerprints
   keep only the narrator's own voice. A readiness report says how many
   usable minutes there are.
4. **It makes the voice,** one of two ways:
   - **Cloned (Qwen3-TTS):** a few clean clips are each tried as the
     reference, and each reads the same sample. You hear the narrator's clip
     beside the clone and pick one. It needs no training, but a clone server
     with a GPU reads the book.
   - **Trained (Piper):** fine-tunes a Piper voice on the recordings, or on
     the cloned voice reading the narrator's sentences (cleaner audio, and as
     much as wanted). Every 1,500 steps it measures how much the voice sounds
     like the narrator and how distorted it is, and keeps a sample to listen
     to. It stops once neither improves. The finished voice is small and
     reads books on any CPU.
5. **You pick by ear.** Nothing goes into Forge's voice list until you choose
   **Use this one**.

> **Use voices responsibly.** Only clone or train a voice you have the right
> to use, and don't publish a voice made from someone's recordings without
> their permission. This repository ships no cloned or trained voices.

---

## Quick start

You need Docker with the Compose plugin. Allow 8 GB of RAM or more and a few
GB of disk for the voice models.

### Any Linux host

```bash
git clone https://github.com/kmmarin/AudioBookForge.git
cd AudioBookForge
./install.sh
```

Then open `http://<host>:8080`. `install.sh` creates `.env` from
`.env.example`, picks the GPU build when it finds an NVIDIA GPU, builds and
starts the app, and downloads the voices and Whisper model.

By hand:

```bash
cp .env.example .env
docker compose up -d --build                                                  # CPU
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build  # NVIDIA GPU
```

To make your audiobooks show up in Audiobookshelf, point `OUTPUT_DIR` in
`.env` at your library.

### Proxmox VE

```bash
deploy/proxmox-deploy.sh root@your-proxmox-host
```

This creates an unprivileged Debian 12 container with Docker, then builds,
starts and prefetches the app. Run it again to update; books, settings and
models are kept. `--help` lists the options (cores, memory, disk, storage,
static IP, VLAN, output folder).

### Windows (WSL 2)

In an admin PowerShell, in the project folder:

```powershell
powershell -ExecutionPolicy Bypass -File deploy\windows-converter.ps1
```

It installs WSL 2 and Ubuntu if they're missing, then Docker inside WSL (and
NVIDIA's container toolkit for the GPU), and starts Forge at
`http://localhost:8080`. If you don't have a server, leave the server
question blank: the PC then keeps its books itself.

### Everything on one machine with a big NVIDIA GPU

A card with 16 GB, such as an RTX 4070 Ti Super, runs all of it:
- Forge on the GPU;
- the voice creator;
- the clone server, for cloned voices and Watch and read's pictures.

**What you need:**
- Windows 11, or Windows 10 22H2, with a recent NVIDIA driver.
- About 80 GB of free disk: the images and models.
- An admin PowerShell.

**Installing:**
1. Unzip the project (or `git clone` it), for example to `C:\audiobook-forge`.
2. In that folder, run:

   ```powershell
   powershell -ExecutionPolicy Bypass -File deploy\windows-converter.ps1 -VoiceCreator -CloneServer -OpenPort
   ```
3. It asks for a server to send books to: leave it blank to keep everything on
   this machine.
4. Open `http://<this PC>:8080`.

**On Linux,** set these in `.env`, then run `./install.sh`:

```ini
VOICELAB=true
CLONE_LOCAL=true
```

The first start downloads several GB. The models for cloned voices (about
7 GB) and pictures (about 7 GB) download the first time each is used.

---

## A server plus a Windows converter

This is the setup the app was built around:
- **The server** runs all the time and is light: it keeps the library, serves
  Read along to every device, and remembers each listener's place.
- **The converter**, a Windows PC with an NVIDIA GPU, voices books many times
  faster and runs the voice creator.

**1. The server.** Install it with `install.sh` or `proxmox-deploy.sh` (above). In its `.env`:

```ini
SHARE_TOKEN=<a long random secret, the same on both machines>
# optional: open the PC's voice creator from the server too
VOICELAB_URL=http://<PC address>:8080
# optional: cloned voices (see below)
CLONE_URL=http://<clone server>:8100
```

Then restart it: `docker compose up -d`.

**2. The converter.** On the Windows PC, in an admin PowerShell, in the project folder:

```powershell
powershell -ExecutionPolicy Bypass -File deploy\windows-converter.ps1 -OpenPort
```

It asks:
- **The server's address**, like `http://192.168.1.10:8080`.
- **Its `SHARE_TOKEN`.** Leave it blank and the script makes one, which you
  then copy to the server.
- **Whether to train voices on this PC.**

`-OpenPort` adds a Windows port proxy and firewall rule so the server can
reach the PC. Give the PC a fixed address (a DHCP reservation). Run the script
again to update; your books, `.env` and models are kept.

**3. Using it.**
- Books you finish on the PC go to the server by themselves (`PUBLISH_AUTO`),
  or when you choose **Send to server**. They arrive with chapters, cover,
  text and the Read along map.
- On the server, changing a book's voice sends the work back to the PC, and
  the book's row shows the PC's progress.
- Your voices are copied to the server, so they can be picked there too.

No Windows? The converter can be any Linux machine with an NVIDIA GPU. Set
`PUBLISH_URL`, `PUBLIC_URL` and `SHARE_TOKEN` in its `.env` by hand.

---

## Making voices

Turn the voice creator on where the NVIDIA GPU is (the converter): answer
yes in `windows-converter.ps1`, or set `VOICELAB=true` in `.env` and deploy
with `deploy/wsl-deploy.sh` (it adds `docker-compose.voice.yml`). Forge's
header then shows **Create a voice**.

Forge and the voice creator take turns on the GPU (`app/gpushare.py`). Voicing
a book always goes first: training pauses, saves its place, and carries on
once Forge is done, so a shared consumer GPU never runs out of memory.

What each kind needs:

| | Cloned (Qwen3-TTS) | Trained (Piper) |
|---|---|---|
| Recordings | a few clean 8–14 s clips (it stops listening after 10 clean minutes) | 15 minutes at least, an hour or more is better |
| Making it | minutes: samples to pick from, no training | hours on the GPU (a GTX 1660 does one to two steps a second) |
| Reading a book | on the clone server's GPU | on any CPU |
| Sound | closest to the narrator | good, can have a slight buzz; learning from the cloned voice helps |

The voice creator's own README has the details: [voicelab/README.md](voicelab/README.md).

---

## Cloned voices: the clone server

Cloned voices are spoken by a small server running
[Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS). It can run on whichever
machine has the GPU for it. Forge (to read books) and the voice creator (to
make samples) reach it at `CLONE_URL`, with the same `CLONE_TOKEN` on both
sides. If `CLONE_TOKEN` is blank, `SHARE_TOKEN` is used.

| Model | Video memory (bfloat16 / float32) |
|---|---|
| Qwen3-TTS 1.7B, the best | about 7 GB / 10 GB |
| Qwen3-TTS 0.6B | about 3 GB / 5.5 GB |

**NVIDIA, with Docker:**

```bash
docker compose -f docker-compose.clone.yml up -d --build
```

On a Windows PC with an NVIDIA GPU, `deploy/clone-setup.sh` does the same
inside WSL.

**AMD Radeon on Windows (no WSL or Docker),** through AMD's ROCm builds of
PyTorch for Windows. On an RX 6800 the 1.7B model reads at about 3.8 times
real time. In PowerShell, from `clone\windows\`:

```powershell
.\setup.ps1                                  # Python, PyTorch for the card and qwen-tts, under D:\forge-clone
Get-Content token.txt | .\save-token.ps1     # the token Forge uses
.\start.ps1                                  # or, in an admin PowerShell: .\install-autostart.ps1 (starts at boot)
```

Use `stop.ps1` to stop it: forcing the process to end while it's working
can leave the GPU driver stuck until a restart.

**Several GPU servers: Settings.** Forge's **Settings** lists every clone
server, whether each is answering and on which GPU, and the jobs each does:
voices, pictures, or both. You can add, change, reorder and remove them there.
- **Who does what:** a job goes to the first server that does it and is
  answering. So one card can speak the voices while another draws the
  pictures, and a second machine can stand in when the first is off.
- **Where it's kept:** the list is stored in `gpu-servers.json` in the models
  volume, shared with the voice creator.
- **The `.env` server:** one set there (`CLONE_URL`) is listed too.
- **Tokens:** they can be entered but are never shown back.

More, including the API and every setting: [clone/README.md](clone/README.md).

---

## HTTPS

```bash
cp https.env.example https.env      # fill in DOMAIN, the DNS API key and SERVER
./https-setup.sh --env https.env
```

It gets a Let's Encrypt certificate through your DNS provider (Cloudflare,
DigitalOcean, Linode, DNSimple or Route 53). Nothing has to be open to the
internet. It then serves the app at `https://<your name>`, and the
certificate renews itself. With Cloudflare it also creates the DNS record.

Other commands:
- `./https-setup.sh --status` shows the certificate and when it expires.
- `./https-setup.sh --off` goes back to plain HTTP.

The API key is kept on the server in `certs/`; `https.env` is in
`.gitignore`.

---

## Configuration

Settings live in `.env`; [.env.example](.env.example) explains every one.
The main ones:

| Setting | Default | What it does |
|---|---|---|
| `PORT` | 8080 | The web port |
| `OUTPUT_DIR` | `./output` | Finished books: point it at your Audiobookshelf library |
| `DEFAULT_FORMAT` | m4b | m4b, m4a, mp3, opus or flac |
| `OUTPUT_LAYOUT` | abs | `abs` (Author/Series/1 - Title/) or `flat` |
| `TTS_WORKERS`, `TORCH_THREADS` | auto | Voice processes and threads each; auto fits your cores and memory |
| `BOOK_CONCURRENCY` | 2 | Books voiced at once |
| `QA_DEFAULT`, `QA_FLAG_WER` | true, 0.12 | The Whisper narration check, and when it flags a chapter |
| `TRANSCRIBE_MODEL` | small | Whisper model for Read along of audiobooks you add (`base` is about 3x faster) |
| `METADATA_LOOKUP` | true | Look books up online |
| `PUBLISH_URL`, `PUBLIC_URL`, `SHARE_TOKEN` | | The server and converter link |
| `VOICELAB`, `VOICELAB_URL` | false | The voice creator, here or on another machine |
| `CLONE_URL`, `CLONE_TOKEN` | | A clone server for cloned voices and pictures (more in Settings) |
| `CLONE_LOCAL` | false | Run the clone server beside Forge on this machine's GPU |
| `SCENE_WORDS`, `DRAW_STYLE` | 75, sketch | Watch and read: words per picture, and the drawing style |
| `INSTALL_CALIBRE` | false | Kindle and other legacy formats (a bigger image) |
| `DEV_FAKE_TTS` | false | Tones instead of speech: tests the whole pipeline in seconds |

The stock voice list is [app/voices.yaml](app/voices.yaml). Add any Kokoro or
Piper voice there, or mount your own copy and set `VOICES_FILE`.

---

## How it works

```
 phones, tablets, browsers
          │
          ▼
 always-on server (CPU)            web + worker, SQLite: the library,
          │       ▲                Read along, each listener's place
 voice    │       │ finished
 changes  │       │ books
          ▼       │
 Windows PC with a GPU (WSL 2)     Forge on CUDA, plus the voice creator
          │                        (they take turns on the GPU)
          │ cloned voices
          ▼
 clone server (any GPU machine)    Qwen3-TTS: NVIDIA in Docker, or AMD on Windows
```

- **web and worker** are one image and two containers. They share state
  through SQLite and never call each other.
- **Voicing:** the worker cuts each chapter into units of about 1,800
  characters, voices them in a shared pool of processes, joins them into
  chapters, and encodes chapters in parallel.
- **Read along:** each unit records the time of every word as it's voiced, so
  the read-along map is built when the book is assembled. For audiobooks you
  add, Whisper's word timings make the same map.
- **The voice creator** is a separate service (`voicelab/`) that Forge passes
  `/createvoice` to. Finished voices go into the shared models volume, and
  Forge lists them without a restart.
- **The clone server** is a separate image (`clone/`), because Qwen3-TTS
  needs an older transformers than Kokoro does.

| Folder | What's there |
|---|---|
| `app/` | Forge: parsing, voicing, assembly, Read along, publishing, the web UI (`app/static/`) |
| `voicelab/` | The voice creator: listening, readiness, cloning, Piper training, quality checks |
| `clone/` | The clone server, and `clone/windows/` for AMD on Windows |
| `deploy/` | Installers: Proxmox LXC, WSL, the Windows converter, the WSL clone server |
| `docker/` | The container entrypoint and the HTTPS proxy |
| `experiments/qwen3-tts/` | The harness used to try Qwen3-TTS before the clone server existed |
| `docs/` | Notes for developers |

---

## Voices and models

Nothing below is in this repository. Each is downloaded from its publisher
the first time it's needed, and is under its own license:

| What | From | License |
|---|---|---|
| Kokoro voices | [hexgrad/Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) | Apache-2.0 |
| Piper voices | [rhasspy/piper-voices](https://huggingface.co/rhasspy/piper-voices) | per voice: see each voice's model card |
| Piper base checkpoints (training) | [rhasspy/piper-checkpoints](https://huggingface.co/datasets/rhasspy/piper-checkpoints) | per voice: see each voice's model card |
| Qwen3-TTS 1.7B and 0.6B Base | [Qwen on Hugging Face](https://huggingface.co/Qwen) | Apache-2.0 |
| Whisper (faster-whisper) | [Systran](https://huggingface.co/Systran) | MIT |
| Speaker fingerprints (ECAPA-TDNN) | [speechbrain/spkrec-ecapa-voxceleb](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb) | Apache-2.0 |

Voices you make stay on your machines (`/models/custom-voices` and
`/models/cloned-voices` in the Docker volume).

---

## Development

```bash
# Forge without any models: tones instead of speech
DEV_FAKE_TTS=true docker compose up -d --build

# the voice creator's tests, from the project root, in the voicelab image
python -m unittest discover -s voicelab/tests

# a running clone server, end to end
PYTHONPATH=. python clone/test_server.py http://host:8100 ref.wav ref.txt TOKEN out/
```

**Branches:**
- `main` is the stable version: install from it.
- `next` collects changes as they're made, with an open pull request into
  `main`. They reach `main` when it's merged.

There's no frontend build step: the UI is plain HTML, CSS and JavaScript in
`app/static/` and `voicelab/lab/static/`. Every setting is an environment
variable read in one `config.py`. [docs/DEVELOPING.md](docs/DEVELOPING.md)
covers the architecture, the API and the conventions.

---

## License

[GPL-3.0](LICENSE). Forge voices books with, and the voice creator trains
with, [piper1-gpl](https://github.com/OHF-Voice/piper1-gpl), which is
GPL-3.0.

It also builds on:
- **Kokoro** (Apache-2.0)
- **qwen-tts** (Apache-2.0)
- **faster-whisper** (MIT)
- **SpeechBrain** (Apache-2.0)
- **PyTorch Lightning** (Apache-2.0)
- **FastAPI** (MIT)
- **ffmpeg** (LGPL/GPL)
- **pandoc** (GPL-2.0+)
- **espeak-ng** (GPL-3.0)
- **Calibre**, optional (GPL-3.0)
