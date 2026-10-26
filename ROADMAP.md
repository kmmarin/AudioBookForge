# Roadmap

What Audiobook Forge does today, what's being worked on, and what's next.
Each open item has an issue for its details and discussion. `main` holds what's
finished; work in progress arrives on `next` first ([README](README.md#development)).

## In progress

| Feature | Where it stands |
|---|---|
| [Watch and read: the first real pictures](https://github.com/kmmarin/AudioBookForge/issues/2) | Scenes, the reader's two views and the drawing endpoint are built. It has drawn with a tiny test model only, not yet on a real GPU. |
| [Cleaner trained (Piper) voices](https://github.com/kmmarin/AudioBookForge/issues/3) | Learning from a cloned voice's audio has cut the buzz from 5–7 dB to 4.0 dB so far. Real recordings measure 3.4. |
| [Kokoro voices made for a narrator](https://github.com/kmmarin/AudioBookForge/issues/4) | It matches the narrator as closely as their own clips do, but distortion is 5.2 dB. Next: steer with a penalty on distortion. |

## Planned

- [Kokoro voice search in the voice creator](https://github.com/kmmarin/AudioBookForge/issues/5)
- [Fine-tune Kokoro on a narrator](https://github.com/kmmarin/AudioBookForge/issues/6)
- [Watch and read: consistent characters and places](https://github.com/kmmarin/AudioBookForge/issues/7)
- [Watch and read: safety and control](https://github.com/kmmarin/AudioBookForge/issues/8)
- [Watch and read: pictures travel with books to the server](https://github.com/kmmarin/AudioBookForge/issues/9)
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
- Trained Piper voices: likeness and distortion measured every 1,500 steps, and samples along the way to pick by ear.
- Cloned voices (Qwen3-TTS): a few clips tried as the reference, a sample each, picked by ear.
- **Learn from the cloned voice:** Piper trained on a cloned voice's clean audio.
- Forge plays Kokoro voicepacks made for a narrator.

**Watch and read**
- Scenes planned from the read-along map, with each one's most paintable sentence as the prompt.
- Pictures drawn by the clone server and given one ink-on-paper look.
- Two views: above the text, or full page with the sentence as a subtitle. Both are adapted from [SeeStory](https://github.com/pgotta/SeeStory) (MIT).

**GPU servers and setup**
- The clone server, for cloned voices and pictures: NVIDIA in Docker or WSL, or natively on Windows (NVIDIA or AMD Radeon).
- A GPU server takes one line to set up on Windows or Linux, and ends by showing the address and token to paste into Forge.
- **Settings → GPU servers:** several servers, each doing voices, pictures or both, with automatic failover. Each has a Test button, and changes are saved with Save.
- Everything is on by default: the voice creator wherever there's an NVIDIA GPU, a clone server beside Forge on a card of 12 GB or more, even loudness and Kindle formats.
- Installers:
  - Linux (`install.sh`)
  - Proxmox LXC
  - a Windows converter (WSL 2)
  - everything on one 16 GB GPU machine
- HTTPS with Let's Encrypt over DNS, with automatic renewal.
