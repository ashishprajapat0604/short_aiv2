# ShortsAI — Hindi Viral Clip Pipeline

Turn a long Hindi video (file or URL) into ready-to-post **9:16 vertical Shorts**.
The AI finds the strongest moments, cuts them clean at full sentences (20–40s each),
and burns in captions exactly the way you want them — or leaves them off.

---

## Contents

- `app.py` — FastAPI server + the web UI route. Orchestrates jobs.
- `select_clips.py` — download → transcribe → AI-select → cut. Writes `clips_manifest.json`.
- `burn_subtitles.py` — per-clip transcription + caption rendering (single ffmpeg pass).
- `index.html` — the web front-end (**must be placed at `templates/index.html`**).
- `requirements.txt` — Python dependencies.
- `fonts/` — **you add this** — holds the Devanagari font (see [Fonts](#fonts-important)).

---

## What you can control

| Feature | Options | Default |
|---|---|---|
| Clip length | 20–40 seconds (min/max overridable) | 20–40s |
| Burn subtitles | on / off (off = clean 9:16, no captions) | on |
| Layout | `single` (one language) · `dual` (Hindi ↑ / English ↓) | single |
| Language *(single)* | `hindi` · `english` · `hinglish` | hindi |
| Position *(single)* | `top` · `middle` · `bottom` (on video) · `below` (under video) | bottom |
| Caption look | `outline` · `box` · `white_box` · `bold_yellow` · `karaoke` · `word_pop` | outline |
| Accent colour | any `#RRGGBB` (used by `karaoke` / `word_pop`) | yellow |
| AI title line | on / off | off |

`english` = translation of the Hindi audio. `hinglish` = the Hindi romanised into
Latin letters. `dual` reproduces the classic Hindi-top / English-bottom look.

**Caption styles:**
- `outline` — clean white text with a black outline (default).
- `box` — white text on a solid dark translucent slab.
- `white_box` — dark text on a solid whitish slab.
- `bold_yellow` — big bold uppercase yellow, heavy outline (classic TikTok look).
- `karaoke` — the phrase stays on screen and the spoken word lights up in the accent
  colour and enlarges (Hormozi-style). Uses word timings for Hindi; for English/Hinglish
  it distributes timing evenly across the words.
- `word_pop` — one big word on screen at a time, scaling/fading in (fast-cut Reels look).

`karaoke` and `word_pop` honour `caption_accent` (`#RRGGBB`) for the highlighted word.

**Caption positioning** (single layout) is geometry-aware. When the source isn't already
9:16 it gets letterboxed (black bars top/bottom), and the position decides where the
caption lands relative to the actual video band:
- `bottom` — ON the video, near its bottom edge (overlaid on the footage; the classic look).
- `below` — just BELOW the video band, in the lower letterbox bar (outside the footage).
- `top` — just ABOVE the video band, in the upper letterbox bar.
- `middle` — centred over the video.

For an already-vertical source with no bars, `below`/`top` fall back to a normal in-frame
margin. The web UI exposes this as a visual picker (tap where the caption should sit).

---

## Requirements

**System (not pip-installable):**
- Python 3.10+
- `ffmpeg` and `ffprobe` on `PATH`
- A **Devanagari font** for Hindi/dual captions (see below)

**Python:** see `requirements.txt`.

```bash
pip install -r requirements.txt
```

Install ffmpeg if needed:
```bash
# Debian / Ubuntu
sudo apt-get update && sudo apt-get install -y ffmpeg
# macOS
brew install ffmpeg
```

---

## Fonts (important)

The earlier "subtitles didn't load" problem was a **missing Devanagari font**. Latin
captions (English/Hinglish) render with Poppins, which most servers already have, but
**Hindi and dual layouts need a Devanagari-capable font**.

Create a `fonts/` folder next to `burn_subtitles.py` and drop a Devanagari `.ttf` in it:

```
fonts/
└── NotoSansDevanagari-Bold.ttf
```

Get it from Google Fonts (Noto Sans Devanagari). The renderer also auto-detects common
system locations (`/usr/share/fonts/.../NotoSansDevanagari*`, Lohit, etc.). If **no**
Devanagari font is found, the logs print a loud `WARNING` and Hindi text may appear as
empty boxes — English/Hinglish/no-caption modes still work fine.

---

## Configuration (`.env`)

Create a `.env` file in the project root:

```ini
# Required — Groq powers Whisper transcription, translation,
# transliteration (Hinglish) and AI titles.
GROQ_API_KEY=your_groq_key

# Optional — per-clip transcription with Deepgram nova-3.
# If omitted, the pipeline reuses the full-video Whisper transcript instead.
DEEPGRAM_API_KEY=your_deepgram_key

# Optional tuning (sensible defaults if unset):
# SUBTITLE_ENGINE=deepgram        # or "whisper" to skip Deepgram
# MAX_RENDER_WORKERS=2            # parallel ffmpeg renders; auto-scales by RAM if unset
# GROQ_SELECTION_MODEL=...        # override the highlight-selection model
# GROQ_SELECTION_MODEL_FALLBACK=...
# SELECTION_CHUNK_CHARS=...       # transcript chunk size for selection
```

---

## Run

```bash
# 1. front-end must live at templates/index.html
mkdir -p templates && cp index.html templates/index.html

# 2. start the server
uvicorn app:app --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000`, paste a link or upload a video, choose your caption
settings, and start. Finished clips appear in the page and download individually or as a zip.

---

## How it works

1. **Select** (`select_clips.py`) — downloads the source (yt-dlp / gdown), transcribes
   the full video with Groq Whisper, asks the LLM for the best moments, snaps each to
   sentence boundaries inside the 20–40s window, and writes `clips_manifest.json`
   (including your `subtitle_options`).
2. **Burn** (`burn_subtitles.py`) — for each clip: transcribes just that clip, runs
   **only** the language step your captions need (translate for English/dual,
   transliterate for Hinglish, titles if enabled), then does a single ffmpeg pass that
   seeks into the source, scales/pads to 1080×1920, and burns the chosen captions.
   With subtitles off, transcription is skipped entirely.

The options travel: **front-end → `app.py` → `select_clips.py` → manifest → `burn_subtitles.py`.**
`app.py` itself needs no changes to support any of the caption settings.

---

## Caption rendering CLI (optional)

You can re-render a job's clips without re-selecting, and override the manifest's
caption settings on the fly:

```bash
# Use whatever is saved in the manifest:
python burn_subtitles.py /path/to/job_dir

# Override on the command line:
python burn_subtitles.py /path/to/job_dir --layout single --language english --position bottom --style karaoke --accent "#2EE640"
python burn_subtitles.py /path/to/job_dir --style word_pop --accent "#FF4FD8"
python burn_subtitles.py /path/to/job_dir --layout dual --title
python burn_subtitles.py /path/to/job_dir --no-burn        # clean clips, no captions
```

Flags: `--no-burn`, `--layout {single,dual}`, `--language {hindi,english,hinglish}`,
`--position {top,middle,bottom,below}`, `--style {outline,box,white_box,bold_yellow,karaoke,word_pop}`,
`--accent "#RRGGBB"`, `--title`, `--manifest PATH`.

---

## Fast iteration — preview captions without the server

`test_captions.py` renders caption styles/positions directly on a local clip, so you
can eyeball changes in seconds instead of running the whole pipeline. It needs **no API
keys** (the Groq/Deepgram SDKs are stubbed and synthetic transcript text is fed in) and
**no server** — you're testing the rendering, which is what changes most.

```bash
# render all 5 styles on one local clip, bottom position
python test_captions.py myclip.mp4

# pick specific styles/positions, choose an accent colour
python test_captions.py myclip.mp4 --styles karaoke,word_pop --positions bottom,top --accent "#2EE640"

# test the dual layout, or your own caption text
python test_captions.py myclip.mp4 --layout dual
python test_captions.py myclip.mp4 --text "Your caption line here"
```

It writes to `caption_preview/`: one `clip_<style>_<pos>.mp4` and `frame_<style>_<pos>.png`
per variant, plus `_contact_sheet.png` — a single labelled grid of every variant so you
can compare looks at a glance. So your loop becomes: **drop in the new `burn_subtitles.py`
→ run this → look.** (`myclip.mp4` can be any short video — one of your already-rendered
Shorts works fine.)

To cut the file-shuffling further:
- Run the server with auto-reload so you don't restart it after each edit:
  `uvicorn app:app --reload`
- Keep the project in git so applying an update is a reviewable diff you can revert.

---



| Method | Path | Purpose |
|---|---|---|
| GET | `/` | Web UI |
| POST | `/process/url` | One-shot: select + burn from a URL |
| POST | `/process/upload` | One-shot: select + burn from an uploaded file |
| POST | `/select-clips/url` · `/select-clips/upload` | Selection only |
| POST | `/burn-subtitles/{job_id}` | Burn an already-selected job |
| GET | `/jobs/{job_id}` | Job status |
| GET | `/jobs/{job_id}/clips` · `/clips.zip` · `/clips/{filename}` | Results |
| GET | `/jobs/{job_id}/highlights` | Selected segments |
| DELETE | `/jobs/{job_id}` | Remove a job (optionally keep files) |
| GET | `/health` | Health check |

Pass caption settings in the `options` object, e.g.:

```json
{
  "num_clips": "auto",
  "min_clip_len": 20,
  "max_clip_len": 40,
  "burn_subtitles": true,
  "subtitle_layout": "single",
  "subtitle_language": "hinglish",
  "subtitle_position": "middle",
  "caption_style": "karaoke",
  "caption_accent": "#FFE600",
  "show_title": false
}
```

---

## Troubleshooting

- **Hindi shows as boxes / blank** → no Devanagari font found. Add one under `fonts/`
  (see [Fonts](#fonts-important)); check the diagnostic log for the `WARNING`.
- **`python-multipart` error on upload** → `pip install python-multipart`.
- **All clips the same length** → expected variety is within 20–40s; widen with
  `min_clip_len`/`max_clip_len` if you want a bigger spread.
- **Deepgram errors** → leave `DEEPGRAM_API_KEY` unset to fall back to the Whisper slice.
- **Per-job logs** → each job writes a `SUBTITLE_DIAGNOSTIC_REPORT.txt` in its job dir.

---

## Notes

- The pipeline assumes **Hindi source audio** (Deepgram language is `hi`). English is a
  translation of it; Hinglish is a romanisation of it.
- `index.html` is served from `templates/index.html` by `app.py`.
