import os
import sys
import math
import uuid
import subprocess
import json
import traceback
import datetime
import gdown
from groq import Groq

# ─────────────────────────────────────────────────────────────
# Diagnostic Logger
# ─────────────────────────────────────────────────────────────

class DiagnosticLog:
    """Writes a human-readable diagnostic report to a .txt file."""

    def __init__(self, job_dir: str):
        self.job_dir = job_dir
        self.path = os.path.join(job_dir, "DIAGNOSTIC_REPORT.txt")
        self.lines = []
        self._write_header()

    def _write_header(self):
        self.lines.append("=" * 70)
        self.lines.append("   CLIP SELECTION DIAGNOSTIC REPORT")
        self.lines.append(f"   Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.lines.append("=" * 70)
        self.lines.append("")

    def section(self, title: str):
        self.lines.append("")
        self.lines.append("-" * 70)
        self.lines.append(f"   {title}")
        self.lines.append("-" * 70)
        self._flush()

    def log(self, msg: str):
        self.lines.append(msg)
        print(msg)
        self._flush()

    def log_json(self, label: str, obj):
        self.lines.append(f"{label}:")
        self.lines.append(json.dumps(obj, indent=2, ensure_ascii=False))
        self._flush()

    def error(self, msg: str, exc: Exception = None):
        self.lines.append(f"[ERROR] {msg}")
        if exc:
            self.lines.append(traceback.format_exc())
        print(f"[ERROR] {msg}", file=sys.stderr)
        self._flush()

    def _flush(self):
        try:
            with open(self.path, "w", encoding="utf-8") as f:
                f.write("\n".join(self.lines))
        except Exception:
            pass

    def finalize(self, raw_clips: list):
        self.section("FINAL RESULT")
        if raw_clips:
            self.log(f"SUCCESS: {len(raw_clips)} raw clip(s) produced:")
            for c in raw_clips:
                path = c["raw_path"]
                size_kb = os.path.getsize(path) // 1024 if os.path.exists(path) else 0
                self.log(f"   - {os.path.basename(path)}  ({size_kb} KB)  score={c.get('score')}  | {c.get('reason','')}")
        else:
            self.log("FAILED: ZERO raw clips were produced. Check errors above.")
        self.log("")
        self.log(f"Report saved to: {self.path}")
        self._flush()


# ─────────────────────────────────────────────────────────────
# Download helpers
# ─────────────────────────────────────────────────────────────

def download_from_gdrive(url: str, output_path: str, log: DiagnosticLog):
    log.log("[Step 1] Google Drive link - downloading via gdown...")
    gdown.download(url, output_path, quiet=False)
    size = os.path.getsize(output_path)
    log.log(f"         Downloaded file size: {size} bytes")
    if size < 100_000:
        with open(output_path, "r", errors="ignore") as f:
            content = f.read(500)
        if "<html" in content.lower():
            raise ValueError(
                "Downloaded an HTML error page instead of video. "
                "Make sure the link is set to 'Anyone with the link can view'."
            )

def _ytdlp_format(quality: str) -> str:
    """Map a UI quality choice to a yt-dlp format string (caps the video height)."""
    q = str(quality or "best").lower()
    heights = {"1080": 1080, "720": 720, "480": 480, "360": 360}
    if q in heights:
        h = heights[q]
        return (f"bestvideo[height<={h}][ext=mp4]+bestaudio[ext=m4a]/"
                f"best[height<={h}][ext=mp4]/best[height<={h}]/best")
    return "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best"


def download_video(url: str, job_dir: str, log: DiagnosticLog, video_quality: str = "best") -> str:
    video_output_path = os.path.join(job_dir, "raw_video.mp4")
    if "drive.google.com" in url:
        download_from_gdrive(url, video_output_path, log)
    else:
        fmt = _ytdlp_format(video_quality)
        log.log(f"[Step 1] Downloading video via yt-dlp...  (quality={video_quality}, format={fmt})")
        command = [
            "yt-dlp",
            "--cookies", "cookies.txt",
            "-f", fmt,
            "--merge-output-format", "mp4",
            "-o", video_output_path,
            url,
        ]
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            log.error(f"yt-dlp failed:\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}")
            raise RuntimeError("yt-dlp download failed")
        log.log(f"         Video saved to: {video_output_path}")
    return video_output_path


# ─────────────────────────────────────────────────────────────
# Audio extraction
# ─────────────────────────────────────────────────────────────

def extract_audio(video_path: str, output_path: str, log: DiagnosticLog) -> str:
    log.log(f"[Audio] Extracting audio from: {video_path}")
    log.log(f"        Video file exists: {os.path.exists(video_path)}")
    log.log(f"        Video file size:   {os.path.getsize(video_path) if os.path.exists(video_path) else 'N/A'} bytes")

    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    command = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-vn", "-ar", "44100", "-ac", "1", "-ab", "64k", "-f", "mp3",
        output_path,
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    log.log(f"        FFmpeg return code: {result.returncode}")
    if result.returncode != 0:
        log.error(f"FFmpeg audio extraction failed:\n{result.stderr}")
        raise RuntimeError(f"FFmpeg audio extraction failed:\n{result.stderr}")
    log.log(f"        Audio saved: {output_path}  ({os.path.getsize(output_path)} bytes)")
    return output_path


# ─────────────────────────────────────────────────────────────
# Full-video transcription (used ONLY for highlight selection)
# ─────────────────────────────────────────────────────────────

def transcribe_full_video_deepgram(audio_path: str, job_dir: str, log: DiagnosticLog) -> str:
    """Calls Deepgram nova-3 on the FULL VIDEO audio once and saves the result as
    transcript_deepgram.json. The burn stage slices this for each clip's subtitles,
    so we only pay for one Deepgram call regardless of how many clips are produced."""
    from deepgram import DeepgramClient
    log.log("[Step 3b] Transcribing full video with Deepgram nova-3 (for subtitle slicing)...")
    api_key = os.environ.get("DEEPGRAM_API_KEY")
    if not api_key:
        log.log("  DEEPGRAM_API_KEY missing — subtitle stage will fall back to per-clip Deepgram calls")
        return None

    client = DeepgramClient()
    with open(audio_path, "rb") as f:
        buf = f.read()

    try:
        response = client.listen.v1.media.transcribe_file(
            request=buf,
            model="nova-3",
            language="hi",
            smart_format=True,
            utterances=True,
        )
        if hasattr(response, "to_dict"):
            data = response.to_dict()
        elif hasattr(response, "model_dump"):
            data = response.model_dump()
        else:
            data = json.loads(response.json())

        # Map Deepgram utterances -> standard segments format (same as burn_subtitles._deepgram_transcribe)
        whisper_format = {"segments": []}
        utterances = data.get("results", {}).get("utterances", [])
        for u in utterances:
            seg = {
                "start": u.get("start"), "end": u.get("end"),
                "text": u.get("transcript"), "words": [],
            }
            for w in u.get("words", []):
                seg["words"].append({
                    "word":  w.get("punctuated_word", w.get("word")),
                    "start": w.get("start"), "end": w.get("end"),
                })
            whisper_format["segments"].append(seg)

        out_path = os.path.join(job_dir, "transcript_deepgram.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(whisper_format, f, indent=2, ensure_ascii=False)
        log.log(f"  Deepgram full-video transcript saved: {out_path} "
                f"({len(whisper_format['segments'])} segments)")
        return out_path
    except Exception as e:
        log.error(f"Deepgram full-video transcription failed: {e}", e)
        return None


def _groq_transcribe_with_retry(client, audio_bytes: bytes, filename: str, log: DiagnosticLog):
    """Call Groq Whisper with retries and model fallback.

    Groq occasionally returns transient 500s and the full 'large-v3' model is more
    prone to them than 'turbo'. We try each model a few times with exponential
    backoff before moving to the next, so one hiccup doesn't kill the whole job."""
    import time
    models = ["whisper-large-v3", "whisper-large-v3-turbo"]
    last_err = None
    for model in models:
        for attempt in range(1, 4):  # 3 attempts per model
            try:
                log.log(f"  Groq transcription: model={model}, attempt {attempt}/3")
                return client.audio.transcriptions.create(
                    file=(filename, audio_bytes),
                    model=model,
                    response_format="verbose_json",
                    language="hi",
                    timestamp_granularities=["word", "segment"],
                )
            except Exception as e:
                last_err = e
                status = getattr(e, "status_code", None)
                # Don't waste retries on auth/permission/bad-request errors.
                if status in (400, 401, 403, 404):
                    log.log(f"  Non-retryable error ({status}) on {model}: {e}")
                    break
                wait = 2 ** attempt  # 2s, 4s, 8s
                log.log(f"  Transcription error on {model} (attempt {attempt}): {e} — retrying in {wait}s")
                time.sleep(wait)
        log.log(f"  Model {model} exhausted; falling back to next model if available.")
    # All models/attempts failed
    raise last_err if last_err else RuntimeError("Groq transcription failed with no error captured")


def transcribe_full_video(audio_path: str, job_dir: str, log: DiagnosticLog) -> str:
    """Transcribe the full video with Groq Whisper for AI highlight selection.
    Tries whisper-large-v3 (better Hindi) with retries, then falls back to
    whisper-large-v3-turbo if the full model keeps erroring."""
    log.log("[Step 3] Transcribing full video with Groq Whisper (for highlight selection)...")
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY is missing from environment!")

    client = Groq(api_key=api_key)

    with open(audio_path, "rb") as f:
        audio_bytes = f.read()
    transcription = _groq_transcribe_with_retry(
        client, audio_bytes, os.path.basename(audio_path), log
    )

    data = json.loads(transcription.model_dump_json())
    segments = data.get("segments", [])
    top_words = data.get("words", [])

    # Map root level word timestamps into segments if nested words list is missing
    if top_words and segments and "words" not in segments[0]:
        for seg in segments:
            seg["words"] = []
        for w in top_words:
            for seg in segments:
                if seg["start"] <= w["start"] <= seg["end"]:
                    seg["words"].append(w)
                    break

    words_total = sum(len(s.get("words", [])) for s in segments)

    log.section("FULL TRANSCRIPTION RESULT")
    log.log(f"  Total segments : {len(segments)}")
    log.log(f"  Total words    : {words_total}")
    if segments:
        log.log(f"  Duration       : {segments[0]['start']:.2f}s to {segments[-1]['end']:.2f}s")

    # Write the human-readable transcript to its OWN plain-text file, so the
    # diagnostic log stays clean and the transcription is easy to read/share.
    transcript_txt_path = os.path.join(job_dir, "transcript_full.txt")
    with open(transcript_txt_path, "w", encoding="utf-8") as tf:
        tf.write("FULL VIDEO TRANSCRIPT\n")
        tf.write(f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        if segments:
            tf.write(f"Duration: {segments[0]['start']:.2f}s to {segments[-1]['end']:.2f}s\n")
        tf.write(f"Segments: {len(segments)}  |  Words: {words_total}\n")
        tf.write("=" * 70 + "\n\n")
        # Timestamped view (one line per segment)
        for seg in segments:
            tf.write(f"[{seg['start']:7.2f} - {seg['end']:7.2f}]  {seg['text'].strip()}\n")
        # Clean prose view (no timestamps), handy for copy/paste
        tf.write("\n" + "=" * 70 + "\nPLAIN TEXT\n" + "=" * 70 + "\n\n")
        tf.write(" ".join(seg["text"].strip() for seg in segments).strip() + "\n")

    # The diagnostic log only points at the transcript file (no giant dump).
    log.log(f"  Transcript text saved: {transcript_txt_path}")

    transcript_path = os.path.join(job_dir, "transcript_full.json")
    with open(transcript_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    log.log(f"  Full transcript JSON saved: {transcript_path}")
    return transcript_path


# ─────────────────────────────────────────────────────────────
# Context-aware boundary snapping
# ─────────────────────────────────────────────────────────────

# Characters that mark the end of a complete spoken thought (Latin + Devanagari danda)
_SENTENCE_END_CHARS = (".", "!", "?", "।", "॥", "…")

# Client requirement: every clip must be at least MIN and at most MAX seconds long.
# These are the defaults; a caller can override per-job via options
# {"min_clip_len": 20, "max_clip_len": 40}. Defined here (before first use as a
# default argument value) so module import doesn't hit a forward reference.
DEFAULT_MIN_CLIP_LEN = 20.0
DEFAULT_MAX_CLIP_LEN = 40.0


def _ends_a_sentence(text: str) -> bool:
    text = (text or "").strip()
    return bool(text) and text[-1] in _SENTENCE_END_CHARS


def snap_to_sentence_boundaries(raw_start: float, raw_end: float, segments: list,
                                 min_dur: float = DEFAULT_MIN_CLIP_LEN,
                                 max_dur: float = DEFAULT_MAX_CLIP_LEN) -> tuple:
    """Snap a raw [start, end] window so the clip begins at the start of a spoken
    thought and ends on a COMPLETE sentence (never mid-context).

    Strategy:
      - Start: snap to the nearest segment start, but prefer one that begins a new
        sentence (i.e. the previous segment ended with sentence punctuation).
      - End: walk forward to the last segment that fits inside max_dur AND ends on
        sentence-ending punctuation. Only if no punctuated end exists do we fall
        back to a plain segment boundary. This is what stops clips being cut in the
        middle of a sentence.
    """
    if not segments:
        return raw_start, raw_end

    # --- choose a start that ideally opens a fresh sentence ---
    idx = min(range(len(segments)), key=lambda i: abs(segments[i]["start"] - raw_start))
    # Nudge to the nearest sentence-opening segment within a small window (<=2 segs).
    for back in range(0, 3):
        j = idx - back
        if j < 0:
            break
        prev_ok = (j == 0) or _ends_a_sentence(segments[j - 1].get("text", ""))
        if prev_ok:
            idx = j
            break
    clip_start = segments[idx]["start"]

    # --- extend the end, preferring a sentence-ending segment within max_dur ---
    last_fit_end = None          # last segment end that fits in max_dur (fallback)
    last_sentence_end = None     # last segment that fits AND ends a sentence (preferred)
    for seg in segments[idx:]:
        if seg["end"] - clip_start > max_dur:
            break
        last_fit_end = seg["end"]
        if seg["end"] - clip_start >= min_dur and _ends_a_sentence(seg.get("text", "")):
            last_sentence_end = seg["end"]

    if last_sentence_end is not None:
        clip_end = last_sentence_end
    elif last_fit_end is not None:
        clip_end = last_fit_end
    else:
        # Single very long segment: clamp to max_dur.
        clip_end = min(segments[idx]["end"], clip_start + max_dur)

    # Guarantee a minimum duration if we can.
    if clip_end - clip_start < min_dur:
        for seg in segments:
            if seg["start"] >= clip_end and seg["end"] - clip_start <= max_dur:
                clip_end = seg["end"]
                if clip_end - clip_start >= min_dur:
                    break

    clip_end = min(clip_end, clip_start + max_dur)
    return round(clip_start, 3), round(clip_end, 3)


# ─────────────────────────────────────────────────────────────
# Clip-count + dense coverage generation
# ─────────────────────────────────────────────────────────────

_ABS_MAX_CLIPS = 80          # hard safety ceiling on clips per video
_AI_MAX = 8                  # how many "best" clips we ask the LLM for
_DEDUP_TOL = 2.0             # clips within 2s start AND 2s end are "identical"


def _length_cycle(min_dur: float, max_dur: float) -> list:
    """Build a small set of varied target lengths that all sit inside
    [min_dur, max_dur], so coverage clips don't all come out the same length."""
    lo, hi = float(min_dur), float(max_dur)
    if hi <= lo:
        return [lo]
    # 4 evenly spaced lengths from min to max (inclusive)
    steps = 4
    return [round(lo + (hi - lo) * i / (steps - 1), 1) for i in range(steps)]


def _distinct(s: float, e: float, chosen: list, tol: float = _DEDUP_TOL) -> bool:
    """True if [s,e] differs from every already-chosen clip by more than tol on
    either the start or the end. Overlap is allowed; near-identical is not."""
    for c in chosen:
        if abs(c["start"] - s) < tol and abs(c["end"] - e) < tol:
            return False
    return True


def _sentence_start_indices(segments: list) -> list:
    """Indices of segments that begin a sentence (previous segment ended on
    punctuation). These make clean, context-safe clip starts."""
    idxs = [i for i, seg in enumerate(segments)
            if i == 0 or _ends_a_sentence(segments[i - 1].get("text", ""))]
    return idxs or list(range(len(segments)))


def _generate_dense_clips(segments: list, n_needed: int, chosen: list,
                          log, min_dur: float = DEFAULT_MIN_CLIP_LEN,
                          max_dur: float = DEFAULT_MAX_CLIP_LEN) -> list:
    """Produce up to n_needed coverage clips SPREAD OUT across the video with minimal
    overlap. Each starts at a sentence boundary, ends on a complete sentence, varies
    in length, and starts at least `min_gap` from every other clip."""
    if n_needed <= 0 or not segments:
        return []

    total = segments[-1]["end"] - segments[0]["start"]
    min_gap = max(min_dur, (total / max(1, n_needed)) * 0.85)

    length_cycle = _length_cycle(min_dur, max_dur)
    starts = _sentence_start_indices(segments)
    pool = []
    for k, si in enumerate(starts):
        a = segments[si]["start"]
        L = length_cycle[k % len(length_cycle)]
        s, e = snap_to_sentence_boundaries(a, a + L, segments, min_dur, max_dur)
        if e - s >= min_dur:
            pool.append((s, e))
    pool.sort()

    picked = []
    chosen_starts = [c["start"] for c in chosen]

    def far_enough(s):
        return all(abs(s - cs) >= min_gap for cs in chosen_starts) and \
               all(abs(s - p[0]) >= min_gap for p in picked)

    for (s, e) in pool:
        if len(picked) >= n_needed:
            break
        if far_enough(s):
            picked.append((s, e))

    # If the gap was too strict to reach the target, relax and top up by even spread.
    if len(picked) < n_needed:
        remaining = [p for p in pool if p not in picked]
        if remaining and n_needed > len(picked):
            stepf = max(1.0, len(remaining) / float(n_needed - len(picked)))
            i = 0.0
            while len(picked) < n_needed and int(i) < len(remaining):
                cand = remaining[int(i)]
                if _distinct(cand[0], cand[1], [{"start": p[0], "end": p[1]} for p in picked] + chosen):
                    picked.append(cand)
                i += stepf

    picked.sort()
    return [{"start": s, "end": e, "score": 6,
             "reason": f"Coverage clip ({e - s:.0f}s)"} for (s, e) in picked]


# ─────────────────────────────────────────────────────────────
# AI selection — MODULAR & CHUNKED
# ─────────────────────────────────────────────────────────────
# The free-tier LLM context is small, so a long transcript is split into chunks
# and each chunk is analysed separately, then the picks are merged. To use a
# paid/large-context model later, set these env vars — NO code change needed:
#   GROQ_SELECTION_MODEL          (e.g. a 128k-context model)
#   GROQ_SELECTION_MODEL_FALLBACK
#   SELECTION_CHUNK_CHARS=200000  (large => whole transcript in ONE call, no chunking)
SELECTION_MODEL_PRIMARY  = os.environ.get("GROQ_SELECTION_MODEL", "llama-3.3-70b-versatile")
SELECTION_MODEL_FALLBACK = os.environ.get("GROQ_SELECTION_MODEL_FALLBACK", "llama-3.1-8b-instant")
SELECTION_CHUNK_CHARS    = int(os.environ.get("SELECTION_CHUNK_CHARS", "6000"))

_SELECTION_PROMPT_HEADER = """You are a short-form content strategist for Instagram/Facebook Reels and YouTube Shorts.
Pick the moments from this transcript portion that would perform best as standalone vertical clips.

WHAT MAKES A CLIP WORK:
1. HOOK IN THE FIRST 3 SECONDS - opens on a bold claim, surprising fact, question, or strong emotion.
2. SELF-CONTAINED - understandable with zero outside context.
3. EMOTIONAL / CONTROVERSIAL PEAK - strong feeling beats neutral info.
4. PAYOFF NEAR THE END - builds to a punchline, revelation, or resolution.
5. ENDS ON A COMPLETE THOUGHT - never stop mid-sentence.

Each clip must be 20-40 seconds and start at the beginning of a sentence."""


def _chunk_segments(segments: list, budget_chars: int) -> list:
    """Split segments into consecutive groups whose combined text stays under
    budget_chars, so each group fits the LLM context window."""
    chunks, cur, cur_len = [], [], 0
    for seg in segments:
        line_len = len(seg.get("text", "")) + 24  # +timestamp overhead
        if cur and cur_len + line_len > budget_chars:
            chunks.append(cur)
            cur, cur_len = [], 0
        cur.append(seg)
        cur_len += line_len
    if cur:
        chunks.append(cur)
    return chunks


def _call_selection_llm(client, prompt: str, log: DiagnosticLog):
    """Single LLM call with model fallback + light retry. Returns parsed list or []."""
    import time
    for model in [SELECTION_MODEL_PRIMARY, SELECTION_MODEL_FALLBACK]:
        for attempt in range(1, 3):
            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.2,
                    response_format={"type": "json_object"},
                )
                raw = resp.choices[0].message.content.strip()
                parsed = json.loads(raw)
                hl = parsed.get("highlights", parsed) if isinstance(parsed, dict) else parsed
                return hl if isinstance(hl, list) else []
            except Exception as e:
                status = getattr(e, "status_code", None)
                if status in (400, 401, 403, 404):
                    log.log(f"    selection model {model}: non-retryable error {status}: {e}")
                    break
                log.log(f"    selection model {model} attempt {attempt} failed: {e}")
                time.sleep(2 * attempt)
    return []


def select_highlights_chunked(segments: list, num_clips: int, per_chunk: int,
                              log: DiagnosticLog) -> list:
    """Run the LLM selector across transcript chunks and merge the picks.
    Returns a list of raw {start, end, score, reason} (un-snapped)."""
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key or not segments:
        return []

    client = Groq(api_key=api_key)
    chunks = _chunk_segments(segments, SELECTION_CHUNK_CHARS)
    log.log(f"  AI selection: {len(segments)} segments -> {len(chunks)} chunk(s) "
            f"(model={SELECTION_MODEL_PRIMARY}, ~{SELECTION_CHUNK_CHARS} chars/chunk, "
            f"{per_chunk} picks/chunk)")

    all_picks = []
    for ci, chunk in enumerate(chunks):
        chunk_text = "".join(f"[{s['start']:.2f} - {s['end']:.2f}] {s['text']}\n" for s in chunk)
        prompt = f"""{_SELECTION_PROMPT_HEADER}

TASK:
From the transcript portion below, return the {per_chunk} strongest clip(s). Use the EXACT
timestamps shown (in seconds). Score each 1-10 (10 = most viral). For "reason", briefly note
the hook and the payoff.

Output ONLY valid JSON: {{"highlights": [{{"start": float, "end": float, "score": int, "reason": "string"}}]}}

TRANSCRIPT PORTION:
{chunk_text}"""
        picks = _call_selection_llm(client, prompt, log)
        log.log(f"    chunk {ci+1}/{len(chunks)}: {len(picks)} pick(s)")
        all_picks.extend(picks)

    return all_picks


# ─────────────────────────────────────────────────────────────
# Highlight Engine (entry point)
# ─────────────────────────────────────────────────────────────

def get_ai_highlights(transcript_path: str, job_dir: str, log: DiagnosticLog,
                      options: dict = None) -> tuple:
    log.section("AI HIGHLIGHT SELECTION")
    api_key = os.environ.get("GROQ_API_KEY")
    highlights_path = os.path.join(job_dir, "highlights.json")

    options = options or {}

    # Read the transcript FIRST so the clip count can scale with video length.
    try:
        with open(transcript_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        segments = data.get("segments", [])
    except Exception as e:
        log.error(f"Failed to read transcript: {e}", e)
        segments = []

    total_duration = (segments[-1]["end"] - segments[0]["start"]) if segments else 60.0
    minutes = max(1, round(total_duration / 60.0))
    log.log(f"  Total video duration: {total_duration:.2f}s (~{minutes} min)")

    # How many clips to produce:
    #   - "auto" (default): one clip per minute of video (15-min video -> ~15 clips).
    #   - an explicit number: used directly, but never more than minutes (1/min cap).
    # Clips are kept relatively non-overlapping; the 1/min cap keeps Deepgram cost
    # (which runs per selected clip later) proportional to the video length.
    max_for_video = max(1, min(minutes, _ABS_MAX_CLIPS))
    raw_nc = options.get("num_clips", "auto")
    auto_mode = bool(options.get("auto_clips")) or str(raw_nc).strip().lower() in ("", "auto", "0", "none")
    if auto_mode:
        num_clips = max_for_video
    else:
        try:
            num_clips = int(raw_nc)
        except (TypeError, ValueError):
            num_clips = max_for_video
            auto_mode = True
        num_clips = max(1, min(num_clips, max_for_video))
    ai_target = num_clips   # let the AI fill as many as it can; coverage tops up the rest
    log.log(f"  Target clips: {num_clips} ({'auto' if auto_mode else 'manual'}, max {max_for_video} = 1/min) | "
            f"AI picks first, coverage fills remainder")

    # Per-clip length bounds (client requirement: 20s-40s by default).
    try:
        min_len = float(options.get("min_clip_len", DEFAULT_MIN_CLIP_LEN))
    except (TypeError, ValueError):
        min_len = DEFAULT_MIN_CLIP_LEN
    try:
        max_len = float(options.get("max_clip_len", DEFAULT_MAX_CLIP_LEN))
    except (TypeError, ValueError):
        max_len = DEFAULT_MAX_CLIP_LEN
    if max_len <= min_len:
        min_len, max_len = DEFAULT_MIN_CLIP_LEN, DEFAULT_MAX_CLIP_LEN
    log.log(f"  Clip length bounds: {min_len:.0f}s min / {max_len:.0f}s max")

    valid = []

    # Run the modular, chunked LLM selector and turn its picks into validated clips.
    per_chunk = max(2, math.ceil(num_clips / max(1, math.ceil(
        (len(segments) and sum(len(s.get('text','')) for s in segments) or 1) / SELECTION_CHUNK_CHARS))))
    raw_picks = select_highlights_chunked(segments, num_clips, per_chunk, log)

    for h in raw_picks:
        try:
            raw_s = float(h.get("start", 0))
            raw_e = float(h.get("end", 0))
        except (TypeError, ValueError):
            continue
        snapped_s, snapped_e = snap_to_sentence_boundaries(raw_s, raw_e, segments, min_len, max_len)
        dur = snapped_e - snapped_s
        if dur >= min_len and _distinct(snapped_s, snapped_e, valid):
            valid.append({
                "start": snapped_s, "end": snapped_e,
                "score": int(h.get("score", 8)) if str(h.get("score", 8)).isdigit() else 8,
                "reason": h.get("reason", "AI-selected highlight"),
            })
    valid.sort(key=lambda v: v.get("score", 0), reverse=True)
    log.log(f"  AI produced {len(valid)} valid, distinct clip(s)")

    # Fill out to the requested count with coverage clips. These may overlap the
    # AI picks and each other, but each is distinct in start point, length, and/or
    # ending (never identical), and always cut on sentence boundaries.
    if len(valid) < num_clips and segments:
        needed = num_clips - len(valid)
        coverage = _generate_dense_clips(segments, needed, valid, log, min_len, max_len)
        log.log(f"\n  Coverage fill: requested {needed} more, generated {len(coverage)}")
        valid.extend(coverage)

    valid = sorted(valid, key=lambda v: v.get("score", 0), reverse=True)[:num_clips]
    log.log(f"\n  Final clip selection ({len(valid)} clips), sorted by predicted performance:")
    for i, v in enumerate(valid):
        log.log(f"    Clip {i+1}: {v['start']:.2f}s -> {v['end']:.2f}s  "
                f"({v['end']-v['start']:.1f}s)  score={v['score']}  | {v['reason']}")

    with open(highlights_path, "w", encoding="utf-8") as f:
        json.dump(valid, f, indent=4, ensure_ascii=False)

    return highlights_path, valid


# ─────────────────────────────────────────────────────────────
# Cut raw clips (NO subtitles burned - that happens in a separate script)
# ─────────────────────────────────────────────────────────────

def cut_raw_clips(video_path: str, highlights: list, job_dir: str, log: DiagnosticLog) -> list:
    """Cuts each highlight range from the source video into its own file,
    without burning subtitles. Returns list of dicts with paths + metadata."""
    log.section("CLIP CUTTING (raw, no subtitles)")
    clips_dir = os.path.join(job_dir, "clips")
    os.makedirs(clips_dir, exist_ok=True)

    log.log(f"   Video path      : {video_path}")
    log.log(f"   Video exists    : {os.path.exists(video_path)}")
    log.log(f"   Total highlights: {len(highlights)}")

    raw_clips = []

    for i, clip in enumerate(highlights):
        start_time = clip["start"]
        end_time = clip["end"]
        clip_index = i + 1
        raw_output = os.path.join(clips_dir, f"viral_clip_{clip_index}_raw.mp4")

        log.log(f"\n   Clip {clip_index}")
        log.log(f"     Range  : {start_time:.3f}s -> {end_time:.3f}s  ({end_time-start_time:.1f}s)")
        log.log(f"     Reason : {clip.get('reason','')}")
        log.log(f"     Output : {raw_output}")

        # Output-seeking (-i before -ss) gives a frame-accurate cut (re-encoded anyway,
        # so no speed penalty from avoiding input-seek). avoid_negative_ts ensures the
        # clip starts at exactly t=0, so a later re-transcription's word timestamps
        # line up 1:1 with the rendered frames (no subtitle drift downstream).
        command = [
            "ffmpeg",
            "-i", video_path,
            "-ss", str(start_time),
            "-to", str(end_time),
            "-avoid_negative_ts", "make_zero",
            "-c:v", "libx264",
            "-preset", "medium",
            "-crf", "18",
            "-c:a", "aac", "-b:a", "192k",
            "-y", raw_output,
        ]
        log.log(f"     FFmpeg cmd: {' '.join(command)}")

        result = subprocess.run(command, capture_output=True, text=True)
        log.log(f"     FFmpeg return code: {result.returncode}")
        if result.returncode != 0:
            log.log(f"     FFMPEG STDERR:\n{result.stderr[-1500:]}")
            log.log(f"     FAILED (non-zero return code)")
            continue

        if not os.path.exists(raw_output):
            log.log(f"     FAILED - output file does not exist")
            continue

        file_size = os.path.getsize(raw_output)
        log.log(f"     Output file size: {file_size} bytes")
        if file_size < 1000:
            log.log(f"     FAILED - output file too small ({file_size} bytes)")
            continue

        log.log(f"     SUCCESS -> {raw_output}")
        raw_clips.append({
            "index": clip_index,
            "raw_path": raw_output,
            "start": start_time,
            "end": end_time,
            "reason": clip.get("reason", ""),
            "score": clip.get("score", 0),
        })

    return raw_clips


# ─────────────────────────────────────────────────────────────
# Main entry point - clip selection only
# ─────────────────────────────────────────────────────────────

def execute_selection_workflow(
    url: str = None,
    local_file_path: str = None,
    options: dict = None,
    status_callback=None,
) -> tuple:
    """Runs download -> transcribe -> AI highlight selection -> cut raw clips.

    Produces a job_dir containing:
      - raw_video.mp4 (or symlinked/copied local file reference)
      - audio.mp3
      - transcript_full.json   (full-video transcript, for reference)
      - highlights.json        (selected highlight ranges + scores + reasons)
      - clips/viral_clip_N_raw.mp4   (cut, no subtitles)
      - clips_manifest.json    (everything a subtitle-burning script needs)
      - DIAGNOSTIC_REPORT.txt

    A second script can point at job_dir and read clips_manifest.json to
    pick up subtitle generation + burning from here.
    """
    job_id  = str(uuid.uuid4())
    job_dir = os.path.join("output", job_id)
    os.makedirs(job_dir, exist_ok=True)

    log = DiagnosticLog(job_dir)
    log.section("JOB INFO")
    log.log(f"   Job ID   : {job_id}")
    log.log(f"   Job dir  : {job_dir}")
    log.log(f"   Input URL: {url or 'N/A'}")
    log.log(f"   Local    : {local_file_path or 'N/A'}")

    if options is None:
        options = {"viral": True, "emotional": True, "key": True, "trend": False, "num_clips": 3}
    options.setdefault("num_clips", 3)
    # Clip-length bounds (client requirement: 20-40s).
    options.setdefault("min_clip_len", DEFAULT_MIN_CLIP_LEN)
    options.setdefault("max_clip_len", DEFAULT_MAX_CLIP_LEN)
    # Subtitle/caption preferences — these don't affect selection, but we persist
    # them into the manifest so the burn stage (burn_subtitles.py) can read them.
    options.setdefault("burn_subtitles", True)
    options.setdefault("subtitle_layout", "single")        # "single" | "dual"
    options.setdefault("subtitle_language", "hindi")        # "hindi" | "english" | "hinglish"
    options.setdefault("subtitle_position", "bottom")       # "top" | "middle" | "bottom"
    options.setdefault("caption_style", "outline")          # outline|box|white_box|bold_yellow|karaoke|word_pop
    options.setdefault("caption_accent", "")                # '#RRGGBB' for karaoke/word_pop active word
    options.setdefault("hindi_font", "")                    # noto|mukta|hind|rozha|kalam ('' = auto)
    options.setdefault("english_font", "")                  # poppins|anton|bebas|archivo|fjalla ('' = auto)
    options.setdefault("video_quality", "best")             # best|1080|720|480|360 (link downloads)
    options.setdefault("show_title", False)
    log.log(f"   Options  : {options}")

    raw_clips = []
    highlights = []

    try:
        if local_file_path:
            if status_callback: status_callback("Step 1/4: Loading local video file...")
            log.section("STEP 1 - VIDEO INPUT")
            log.log(f"   Using local file: {local_file_path}")
            log.log(f"   File exists      : {os.path.exists(local_file_path)}")
            log.log(f"   File size        : {os.path.getsize(local_file_path) if os.path.exists(local_file_path) else 'N/A'} bytes")
            video_path = local_file_path
        else:
            if status_callback: status_callback("Step 1/4: Downloading video...")
            log.section("STEP 1 - VIDEO DOWNLOAD")
            video_path = download_video(url, job_dir, log, options.get("video_quality", "best"))

        if status_callback: status_callback("Step 2/4: Extracting audio...")
        log.section("STEP 2 - AUDIO EXTRACTION")
        full_audio_path = extract_audio(video_path, os.path.join(job_dir, "audio.mp3"), log)

        # ── STAGE 2: ONE Whisper transcription of the full video (for selection only).
        # We deliberately do NOT run Deepgram on the whole video — that bills the full
        # duration. Deepgram runs later, per selected clip only (much cheaper here).
        if status_callback: status_callback("Step 3/4: Transcribing full video (Whisper)...")
        log.section("STEP 3 - FULL TRANSCRIPTION")
        transcript_path = transcribe_full_video(full_audio_path, job_dir, log)

        # ── STAGE 3: AI clip selection (chunked LLM; no video cutting here) ──
        if status_callback: status_callback("Step 4/4: AI is finding the most engaging moments...")
        _, highlights = get_ai_highlights(transcript_path, job_dir, log, options)

        # NOTE: We do NOT cut raw clips anymore. The burn stage seeks into the source
        # video directly (-ss/-to) and cuts + scales + burns in a single ffmpeg pass,
        # which removes an entire encode/IO round-trip per clip.
        clip_entries = [{
            "index": i + 1,
            "raw_path": video_path,      # burn stage seeks into the source video
            "start": h["start"],
            "end": h["end"],
            "reason": h.get("reason", ""),
            "score": h.get("score", 0),
        } for i, h in enumerate(highlights)]
        raw_clips = clip_entries

        # Write manifest for the next (subtitle) script
        manifest = {
            "job_id": job_id,
            "job_dir": job_dir,
            "video_path": video_path,
            "audio_path": full_audio_path,
            "transcript_path": transcript_path,
            "source_is_full_video": True,   # raw_path points at the full source
            # Caption/subtitle preferences chosen by the client, read by burn_subtitles.py.
            "subtitle_options": {
                "burn_subtitles":    options.get("burn_subtitles", True),
                "subtitle_layout":   options.get("subtitle_layout", "single"),
                "subtitle_language": options.get("subtitle_language", "hindi"),
                "subtitle_position": options.get("subtitle_position", "bottom"),
                "caption_style":     options.get("caption_style", "outline"),
                "caption_accent":    options.get("caption_accent", ""),
                "hindi_font":        options.get("hindi_font", ""),
                "english_font":      options.get("english_font", ""),
                "show_title":        options.get("show_title", False),
            },
            "clips": clip_entries,
        }
        manifest_path = os.path.join(job_dir, "clips_manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=4, ensure_ascii=False)
        log.log(f"\nManifest saved: {manifest_path}")

    except Exception as e:
        log.section("SELECTION PIPELINE CRASHED")
        log.error(f"Unhandled exception: {e}", e)
        raw_clips = []
        highlights = []

    log.finalize(raw_clips)

    return raw_clips, highlights, log.path
