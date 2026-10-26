# Voice creator

Makes a narrator's voice from recordings of them (a few minutes to a whole
audiobook) and puts it in Forge's voice list, once you've listened and picked
it. It's a service of its own (`voicelab`), reached through Forge at
`/createvoice`. A voice is made one of two ways:

- **Cloned:** Qwen3-TTS reads in the narrator's voice from one clip of them,
  with no training. A clone server with a GPU reads the book
  ([clone/README.md](../clone/README.md)).
- **Trained:** a Piper voice fine-tuned on this machine's GPU for hours. The
  finished voice reads books on any CPU.

## Turning it on

It's on by default wherever there's an NVIDIA GPU: `deploy/windows-converter.ps1`,
`deploy/wsl-deploy.sh` and `install.sh` add it there (`VOICELAB=false` in `.env`
leaves it out). Cloned voices also need a GPU server: Forge's **Settings → GPU
servers** shows how to set one up. By hand:
`docker compose -f docker-compose.yml -f docker-compose.gpu.yml -f docker-compose.voice.yml up -d --build`.
Forge's header then shows **Create a voice**.

On a server without a GPU, point Forge at the PC's Forge:

```
VOICELAB_URL=http://<PC address>:8080
```

The server then shows **Create a voice** too and passes `/createvoice` on to
the PC, uploads and live progress included. When the PC is off it shows a
page saying so, and opens the voice creator by itself once the PC answers.
With HTTPS, nginx passes `/createvoice/` with no upload size limit.

## How it works

1. **Upload** (`lab/main.py`): in 16 MB parts, streamed to disk. Adding the
   same file again carries on an interrupted upload, and a file that's the
   same audio as one already added (a sync tool's "conflicted copy") is
   skipped.
2. **Name:** typed by you, or found. It looks in the tags (narrator, `©nrt`,
   composer, performer, "Narrated by …"), then for "read by …" in the opening
   minutes of the first part, then at the file name.
3. **Listening** (`lab/analyze.py`): five minutes at a time, never the whole
   file in memory. faster-whisper with word timings cuts 1–15 s clips at
   pauses, and each clip is checked for music, background level, clipping and
   Whisper's confidence. Progress is saved after every window. It stops once
   enough clean speech is heard (`LISTEN_ENOUGH_MINUTES`; only
   `LISTEN_CLONE_MINUTES` for a voice to be cloned).
4. **Readiness** (`lab/readiness.py`):
   - speech heard twice is skipped;
   - only the narrator's own voice is kept (ECAPA speaker fingerprints
     against the main voice), which drops intros, announcers, other speakers
     and strong character voices;
   - credit chapters are left out.

   The report gives usable minutes, where every minute went, sound coverage,
   speaker consistency and a rating.

   **Pick the clips** (optional): narrators do character voices, and some
   slip past the speaker check while some of the narrator's own lines are
   dropped. The card lists the clips least like the narrator first (or in
   book order) to play and approve or leave out, by mouse or keyboard
   (Space plays, A approves, X leaves out).
   - With **Only use the clips I approve** on, training and its held-out check use only approved clips. Every recording is then listened to, so there's all of it to choose from.
   - Approving a clip overrides the speaker check, but not music, noise, clipping, credits or a duplicate.
   - With it off, nothing changes; the picks wait.
5. **Making it**, with the model picked for the project (a Qwen3-TTS model,
   or Piper):
   - **Cloned** (`lab/clone.py`): a few clean 8–14 s clips are tried as the
     reference. The clearest are chosen, preferring the narrator's usual voice
     to a character's, and different recordings and stretches of them. Each
     reads the same sample passage. Its likeness to the narrator is measured,
     and you hear the narrator's clip beside the clone.
   - **Trained** (`lab/train.py`): piper1-gpl's own VITS code fine-tunes the
     stock voice for the accent that already sounds most like the narrator.
     Every 1,500 steps the voice so far is exported (on the CPU) and says
     held-out sentences. It's measured for likeness (speaker fingerprints
     against the narrator's real clips) and distortion (tonal buzz in the
     spectrum, `quality.artifacts`), and reads the sample passage for you to
     hear. The first round lasts `TRAIN_FIRST_MINUTES`; **Train further**
     carries on until neither number improves. Training resumes after a
     restart.
   - **Learn from the cloned voice:** a Piper round whose training audio is
     the project's cloned voice reading the narrator's own sentences
     (`TEACH_MINUTES` of them, made by the clone server), instead of clips cut
     from the recordings. That audio says exactly its text, in one steady
     voice, and there can be more of it than the recordings hold. The
     narrator's real held-out clips stay the measure.
6. **You pick by ear:** **Use this one** on a sample or a training check puts
   that voice into Forge. Nothing goes in by itself.
   - A trained voice is `<name>.onnx` + `.onnx.json` in `/models/custom-voices`.
   - A cloned voice is a folder in `/models/cloned-voices`: `ref.wav`, `ref.txt`, `voice.json`.

   Forge lists either without a restart, and sends them on to a server Forge.
7. **Quality report** (`lab/quality.py`, for trained voices): the held-out
   clips (never trained on) are said by the new voice and by the starting
   voice. It gives clarity (Whisper's word error rate), likeness and
   distortion, with the narrator's own clips as the yardstick, and plays the
   same sentences side by side.

## Sharing the GPU

Forge's worker and the voice creator take turns (`app/gpushare.py`): each
writes a small state file with a heartbeat in `/models/gpu-share`.
- **Forge goes first.** Training (and listening on the GPU) waits while Forge
  is voicing. Forge finishes what it has in flight, starts no new books, and
  shuts its voice processes down, so their video memory is really freed.
- **A book interrupts training.** If one arrives while training holds the GPU,
  training saves its place, lets go, and carries on afterwards.
- **Memory is checked first.** Before training, free video memory is checked
  with nvidia-smi, and PyTorch is capped below it.
- **Cloned voices don't take part.** They run on the clone server, so they
  never wait for this GPU.

## Settings

See the voice creator and cloned voices sections of `.env.example`, and
`lab/config.py` for the clip filters, readiness thresholds, training limits
and likeness checks.

## Tests

```
python -m unittest discover -s voicelab/tests     # from the project root, in the voicelab image
```

## Licences

- **piper1-gpl** (GPL-3.0): its training code is copied into the image.
- **Base checkpoints** from `rhasspy/piper-checkpoints`: each voice's license
  is on its model card (en_US lessac: Blizzard 2013 data, research and
  personal use; en_GB alba: CC BY 4.0).
- **qwen-tts and the Qwen3-TTS models**, through the clone server (Apache-2.0).
- **faster-whisper** (MIT) with Whisper (MIT).
- **SpeechBrain ECAPA-TDNN** (Apache-2.0, trained on VoxCeleb).
- **PyTorch Lightning** (Apache-2.0).

Only make voices of people whose recordings you have the right to use, and
don't share a voice made from someone's recordings without their permission.
