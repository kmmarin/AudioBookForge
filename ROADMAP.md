# Roadmap

What Audiobook Forge does today, what's being worked on, and what's next.
Each open item has an issue for its details and discussion. `main` holds what's
finished; work in progress arrives on `next` first ([README](README.md#development)).

## In progress

| Feature | Where it stands |
|---|---|
| [Cleaner trained (Piper) voices](https://github.com/kmmarin/AudioBookForge/issues/3) | Learning from a cloned voice's audio has cut the buzz from 5–7 dB to 4.0 dB so far. Real recordings measure 3.4. |
| [Kokoro voices made for a narrator](https://github.com/kmmarin/AudioBookForge/issues/4) | It matches the narrator as closely as their own clips do, but distortion is 5.2 dB. Next: steer with a penalty on distortion. |

## Planned

- [Kokoro voice search in the voice creator](https://github.com/kmmarin/AudioBookForge/issues/5)
- [Fine-tune Kokoro on a narrator](https://github.com/kmmarin/AudioBookForge/issues/6)
- [Move books and voices between machines](https://github.com/kmmarin/AudioBookForge/issues/10)
- [Tests on every push](https://github.com/kmmarin/AudioBookForge/issues/11)

## Implemented

**Books to audiobooks**
- EPUB, PDF, Word, plain text and more in, with chapters found and front matter turned off.
- Book details from the file and online (Open Library, Audible, Google Books), with covers.
- Kokoro and Piper voices: per-book voice, mood and speed, and **Compare voices** with a saved 10-second sample each.
- M4B, M4A, MP3, Opus or FLAC out, with chapters, tags and cover, in an Audiobookshelf layout.
- Fast: chapters voiced in parallel units on CPU or NVIDIA GPU, cached, so an edit re-voices only what changed.
- The Whisper narration check, and **Redo** for chapters it flags.
- Every voice a book is made in is kept; switching between them takes seconds.
- Exports, under a small **⋯**: a book's audio, its text, and a voice you made, as a zip.
- **Find a free book:** Project Gutenberg's public-domain books, searched in Forge and added with a click, within their terms of use.
- A book not voiced yet can be read in the reader as text.

**Read along**
- Word-by-word highlighting, in any browser.
- Listeners, each with their own place and checkpoints on every device.
- Sleep timer and chapter skips.
- Playback speed with − and +, and the time left shown at that speed.
- Audiobooks you already have: Whisper writes down their text with word timings, they're imported into the library as they are, and audiobooks in parts are combined into one M4B.

**Two machines**
- An always-on server plus a GPU converter.
- Books sent over automatically.
- A voice change asked for on the server is voiced on the converter, with live progress.
- **Voice it here instead** when the converter is off.

**Making voices** (the voice creator)
- Recordings heard, cleaned and checked: speech, music, noise and other speakers are told apart, with a readiness report.
- **Pick the clips:** listen and approve the narrator's own clips; training and its quality check can use only those.
- Trained Piper voices: likeness and distortion measured every 1,500 steps, and samples along the way to pick by ear.
- Cloned voices (Qwen3-TTS): a few clips tried as the reference, a sample each, picked by ear.
- **Learn from the cloned voice:** Piper trained on a cloned voice's clean audio.
- Forge plays Kokoro voicepacks made for a narrator.

**GPU servers and setup**
- The clone server, for cloned voices: NVIDIA in Docker or WSL, or natively on Windows (NVIDIA or AMD Radeon).
- A GPU server takes one line to set up on Windows or Linux, and ends by showing the address and token to paste into Forge.
- **Settings → GPU servers:** several servers, with automatic failover. Each has a Test button, and changes are saved with Save.
- Every card in a GPU server works at once, each benchmarked before its first job so it gets as much work as it can take; Settings shows the results, chooses the cards and benchmarks again.
- Cloned voices fall back to this machine's own card when the model fits it.
- Everything is on by default: the voice creator wherever there's an NVIDIA GPU, a clone server beside Forge on a card of 12 GB or more, even loudness and Kindle formats.
- Installers:
  - Linux (`install.sh`)
  - Proxmox LXC
  - a Windows converter (WSL 2)
  - everything on one 16 GB GPU machine
- HTTPS with Let's Encrypt over DNS, with automatic renewal.
