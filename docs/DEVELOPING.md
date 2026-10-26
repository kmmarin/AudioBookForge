# Developing Audiobook Forge

How the code fits together, the API, and the conventions to keep. The
[README](../README.md) covers what it does and how to run it.

## Architecture

```
docker-compose.yml
  web     FastAPI + static UI (uvicorn, port 8000 -> host PORT, default 8080)
  worker  python -m app.worker (the same image, another command)
  volumes ./data -> /data (SQLite + book files), ./output -> /output, "models" -> /models
docker-compose.gpu.yml     NVIDIA: the CUDA build of the same image
docker-compose.voice.yml   + voicelab, the voice creator (its own image), at /createvoice
docker-compose.clone.yml   the clone server alone (its own image), for any machine with a GPU
docker-compose.https.yml   + certbot (DNS challenge) and nginx in front
```

The web app and the worker share state through SQLite (`/data/forge.db`, WAL
mode) and never call each other: the web app sets `status='queued'`, the
worker polls for it. The voice creator has its own database in its data
folder; what it shares with Forge is the models volume (finished voices, the
GPU turn files, Whisper).

### Book lifecycle

`parsing -> ready -> queued -> voicing -> assembling -> checking -> done`, with
`failed` and `cancelled` on the side; `done`, `failed` and `cancelled` can be
started again, reusing cached audio. An audiobook added for Read along
(`kind = 'audio'`) goes `parsing -> ready -> queued -> transcribing -> done`.

### File map

| File | Purpose |
|---|---|
| `app/config.py` | Every setting, from environment variables. Sizes workers from the cgroup's CPU and memory limits. |
| `app/db.py` | SQLite schema and helpers: books, chapters, samples, listeners, progress, checkpoints. |
| `app/parsers.py` | Format routing, chapter detection, metadata (see Parsing below). |
| `app/audiofile.py` | Audiobooks added for Read along: tags, cover and chapters from ffprobe. |
| `app/lookup.py` | Book details online (Open Library, Audible, Google Books), scored against the file. |
| `app/textprep.py` | Text cleanup for speech, spoken chapter titles, cutting chapters into units, unit hash keys. |
| `app/tts.py` | Kokoro, Piper, clone and fake engines, run inside worker processes. `synth_unit(job)` writes one WAV and its word timings. |
| `app/clone.py` | The clone server's client (its API is described at the top). |
| `app/worker.py` | Scheduler: the shared process pool, clone threads, progress, cancellation, crash recovery, assembly, the narration check, saved samples. |
| `app/assemble.py` | Joins units into chapters, encodes chapters in parallel, writes chapter markers, tags and cover, builds the output path. |
| `app/qa.py` | The narration check: faster-whisper on a sample of units, word error rate via jiwer. |
| `app/readalong.py` | The read-along map (where every word is in the audio), built at assembly. |
| `app/transcribe.py`, `app/transcript.py` | Read along for added audiobooks: Whisper with word timings, a chapter at a time; the text and map from it. |
| `app/versions.py` | Kept copies of a book in every voice it was made in. |
| `app/listeners.py` | Listeners: each person's place and checkpoints per book. |
| `app/publish.py` | Sending finished books and your voices to a server Forge. |
| `app/previews.py` | Voice samples, and the saved 10-second one per voice. |
| `app/gpushare.py` | Forge and the voice creator taking turns on one GPU (the handshake is described at the top). |
| `app/covers.py` | Cover uploads normalized to JPEG; a plain cover drawn when a book has none. |
| `app/voices.yaml`, `app/voices.py` | The stock voices, plus the ones made with the voice creator. |
| `app/main.py` | The REST API, the server-sent event stream and the `/createvoice` pass-through. |
| `app/static/` | Plain JS UI, no build step: `index.html`/`app.js` (the library), `read.html`/`read.js` (Read along). |
| `voicelab/lab/` | The voice creator: `analyze.py` (listening), `readiness.py`, `clone.py`, `dataset.py`, `train.py`, `quality.py`, `voices.py` (installing into Forge), `jobs.py` (lanes of jobs). |
| `clone/server.py` | The clone server: Qwen3-TTS, batched, one model loaded at a time. |

### API (all under `/api`)

- `GET health`, `GET config` (voices, formats, accepted extensions)
- `GET books`, `GET books/{id}`, `POST books` (multipart, field `files`, several at once)
- `PATCH books/{id}` `{meta, voice, mood, speed, fmt, qa, chapters: [{idx, title, include}]}`
- `POST bulk` `{ids, voice?, fmt?, speed?, qa?}`, `POST start` `{ids}`
- `POST books/{id}/cancel`, `POST books/{id}/redo`, `POST books/{id}/rebuild`, `DELETE books/{id}`
- `GET`/`PUT books/{id}/chapters/{idx}/text`
- `GET`/`POST books/{id}/cover`, `POST books/{id}/cover/fetch`, `GET books/{id}/lookup`, `POST books/{id}/match`
- `GET books/{id}/file` (download), `GET books/{id}/audio` (streaming, with ranges), `GET books/{id}/readalong`
- `GET`/`PUT books/{id}/progress`, `GET`/`POST books/{id}/checkpoints`, `DELETE books/{id}/checkpoints/{cid}`;
  these take `?listener=<id>`
- `PATCH`/`DELETE books/{id}/versions/{vid}`, `POST books/{id}/versions/{vid}/use`
- `POST books/{id}/publish`, `POST import` (a book from another Forge), `POST`/`GET books/{id}/revoice`
- `GET`/`POST`/`DELETE custom-voices/...` and `cloned-voices/...` (voice sync between Forges, token checked)
- `GET preview?voice=&speed=&seconds=`, `GET books/{id}/sample`
- `GET events`: server-sent events, the whole book list whenever it changes

Forge to Forge calls carry `X-Forge-Token: <SHARE_TOKEN>`.

### On disk, per book

`/data/books/<id>/`:
- `source.<ext>`: the uploaded file
- `text/NNNN.txt`: one per chapter, editable
- `cover.jpg` (or `cover-auto.jpg`)
- `units/CCCC-UUUU-<sha1>.wav`
- `readalong.json`, `transcript/` (added audiobooks)
- `voices/<version>/`: kept versions

## Speed design

- **Units:** each chapter is cut into units of about `UNIT_CHARS=1800`
  characters at paragraph or sentence boundaries. Units from every active book
  go into one `ProcessPoolExecutor` (spawn), longest first, a few at a time,
  so books take turns and voice samples go next.
- **One model per process:** each worker process loads its own model, which
  sidesteps the GIL. `TTS_WORKERS` × `TORCH_THREADS` is sized from cores and
  memory.
- **Cached by content:** unit file names hash the voice, speed and text, so
  restarts, cancels and text edits only re-voice what changed.
- **Assembly:** units are joined into chapter WAVs, and chapters are encoded
  in parallel (`ENCODE_WORKERS`). M4B, M4A and MP3 are joined with ffmpeg's
  concat demuxer and `-c copy`; Opus and FLAC do one final encode.
- **The narration check** runs alongside assembly, on `QA_SAMPLE` of each
  chapter's units, against the units' own text (no alignment needed).
- **Cloned voices** go to the clone server through their own threads, a
  unit's sentences in one request, so they never wait for this machine's GPU.

## Parsing

| Input | Method | Chapters from |
|---|---|---|
| `.epub` | ebooklib + BeautifulSoup | The table of contents mapped onto the spine; footnotes, `<sup>` and tables removed; drop caps handled; series from Calibre or EPUB 3 metadata. |
| `.pdf` | pypdf | Bookmarks, else heading detection; repeated headers, footers and page numbers removed; hyphenated line breaks joined. A scan with no text layer fails with an OCR hint. |
| `.txt` | Heading patterns | `CHAPTER IV` and a subtitle, "Part One", prologue and epilogue, bare numerals, Markdown `#`; Project Gutenberg header and footer stripped. Falls back to 4,500-word parts. |
| docx, odt, rtf, html, md, fb2, rst, org, tex… | pandoc to text with heading markers | The heading level with the most occurrences; repeating higher levels become title prefixes. |
| mobi, azw3 and other Kindle formats | Calibre's `ebook-convert` to EPUB | With `INSTALL_CALIBRE=true`; not DRM-protected files. |
| `.doc` | antiword | As plain text. |

Missing details are then filled from the file name (`Author - Title (Series #2)`,
`Title by Author`), and front and back matter is turned off.

## Testing

```bash
# Forge without models: DEV_FAKE_TTS=true makes tones instead of speech
pip install fastapi "uvicorn[standard]" python-multipart pyyaml numpy soundfile ebooklib \
  beautifulsoup4 lxml pypdf charset-normalizer pillow jiwer
export DATA_DIR=/tmp/fd OUTPUT_DIR=/tmp/fo MODELS_DIR=/tmp/fm DEV_FAKE_TTS=true TTS_WORKERS=3
uvicorn app.main:app --port 8765 &        # needs ffmpeg, ffprobe and pandoc on PATH
python -m app.worker &
curl -F "files=@book.epub" http://127.0.0.1:8765/api/books
curl -X POST -H 'Content-Type: application/json' -d '{"ids":["<id>"]}' http://127.0.0.1:8765/api/start

# The voice creator's unit tests (in the voicelab image, from the project root)
python -m unittest discover -s voicelab/tests

# A running clone server, end to end
PYTHONPATH=. python clone/test_server.py URL REF.wav REF.txt [TOKEN] [OUT_DIR]
```

## Conventions

- **UI copy:**
  - sentence case and plain verbs;
  - buttons named for what they do ("Start", "Download", "Save and rebuild");
  - errors say what happened and how to fix it;
  - no all-caps labels;
  - typographic apostrophes and quotes (’ “ ”) in what people read.
- **Design:**
  - cool grey surfaces with one amber accent (`--signal`);
  - green for done, red for faults;
  - light and dark themes both come from CSS variables;
  - the per-book chapter meter is the signature element: bar height is
    chapter length, the fill is voicing progress, hatched bars are skipped
    chapters, and a red cap marks a misread chapter.
- **Code:**
  - every setting is an environment variable read in `config.py`;
  - heavy imports (torch, kokoro, piper, faster_whisper) stay lazy so the web
    container starts fast;
  - worker code must be safe to restart at any moment;
  - nothing goes into Forge's voice list until a person picks it by ear.
- **Portability:**
  - no frontend build step, no Redis, no second database;
  - one image runs as two services, sharing SQLite.
