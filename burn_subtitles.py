import os
import sys
import re
import math
import json
import shutil
import tempfile
import traceback
import datetime
import subprocess
import argparse
from groq import Groq
# Note: PrerecordedOptions is removed in Deepgram SDK v5.0.0+
from deepgram import DeepgramClient
import providers

# ─────────────────────────────────────────────────────────────
# Diagnostic Logger
# ─────────────────────────────────────────────────────────────

class DiagnosticLog:
    """Writes a human-readable diagnostic report to a .txt file."""

    def __init__(self, job_dir: str):
        self.job_dir = job_dir
        self.path = os.path.join(job_dir, "SUBTITLE_DIAGNOSTIC_REPORT.txt")
        self.lines = []
        self._write_header()

    def _write_header(self):
        self.lines.append("=" * 70)
        self.lines.append("   SUBTITLE BURNING DIAGNOSTIC REPORT")
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

    def finalize(self, final_clips: list):
        self.section("FINAL RESULT")
        if final_clips:
            self.log(f"SUCCESS: {len(final_clips)} subtitled clip(s) produced:")
            for c in final_clips:
                size_kb = os.path.getsize(c) // 1024 if os.path.exists(c) else 0
                self.log(f"   - {os.path.basename(c)}  ({size_kb} KB)")
        else:
            self.log("FAILED: ZERO subtitled clips were produced. Check errors above.")
        self.log("")
        self.log(f"Report saved to: {self.path}")
        self._flush()


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


def _extract_clip_audio(source_video: str, output_path: str,
                        start: float, end: float, log: DiagnosticLog) -> str:
    """Extract ONLY the [start, end] slice of audio from the source video to mp3,
    16kHz mono (small + ideal for speech recognition). Timestamps in the result are
    clip-local (0 = clip start) because we seek before -i."""
    if not os.path.exists(source_video):
        raise FileNotFoundError(f"Source video not found: {source_video}")
    cmd = ["ffmpeg", "-y"]
    if start is not None and end is not None:
        cmd += ["-ss", f"{float(start):.3f}", "-to", f"{float(end):.3f}"]
    cmd += ["-i", source_video, "-vn", "-ar", "16000", "-ac", "1", "-f", "mp3", output_path]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Clip audio extraction failed:\n{result.stderr[-800:]}")
    return output_path


# ─────────────────────────────────────────────────────────────
# Per-clip Deepgram transcription (SDK v5.0.0+)
# ─────────────────────────────────────────────────────────────

def _deepgram_transcribe(audio_path: str, log: DiagnosticLog) -> dict:
    """Calls Deepgram nova-3 on the given audio file and returns parsed JSON
    mapped to the exact same 'segments' + 'words' structure expected by the pipeline."""
    api_key = os.environ.get("DEEPGRAM_API_KEY")
    if not api_key:
        raise ValueError("DEEPGRAM_API_KEY is missing from environment!")

    client = DeepgramClient(api_key=api_key)

    with open(audio_path, "rb") as f:
        buffer_data = f.read()

    # V5.0.0 API Call format
    # nova-3 has significantly better multilingual (incl. Hindi) accuracy than nova-2
    response = client.listen.v1.media.transcribe_file(
        request=buffer_data,
        model="nova-3",
        language="hi",
        smart_format=True,
        utterances=True,
    )
    
    # FIX: Safely convert the Pydantic response object to a dictionary
    if hasattr(response, "to_dict"):
        data = response.to_dict()
    elif hasattr(response, "model_dump"):
        data = response.model_dump() # Pydantic v2
    else:
        data = json.loads(response.json()) # Pydantic v1

    # Map Deepgram's 'utterances' into the standard Whisper 'segments' format
    whisper_format = {"segments": []}
    utterances = data.get("results", {}).get("utterances", [])

    for u in utterances:
        segment = {
            "start": u.get("start"),
            "end": u.get("end"),
            "text": u.get("transcript"),
            "words": []
        }
        
        # Prefer the punctuated word if available for cleaner subtitles
        for w in u.get("words", []):
            segment["words"].append({
                "word": w.get("punctuated_word", w.get("word")),
                "start": w.get("start"),
                "end": w.get("end")
            })
            
        whisper_format["segments"].append(segment)

    # Fallback mapper just in case utterances array is empty but raw words exist
    if not whisper_format["segments"]:
        channels = data.get("results", {}).get("channels", [])
        if channels and channels[0].get("alternatives"):
            alt = channels[0]["alternatives"][0]
            words = alt.get("words", [])
            if words:
                whisper_format["segments"].append({
                    "start": words[0].get("start"),
                    "end": words[-1].get("end"),
                    "text": alt.get("transcript"),
                    "words": [{"word": w.get("punctuated_word", w.get("word")), "start": w.get("start"), "end": w.get("end")} for w in words]
                })

    return whisper_format


def _slice_transcript(full_transcript_path: str, clip_start: float, clip_end: float) -> dict:
    """Slice the full-video Deepgram transcript JSON to the clip's [start, end] window
    and shift all timestamps to clip-local time (0 = clip start).

    This avoids re-calling Deepgram for every overlapping clip — the single full-video
    transcription already has every word with precise timestamps; we just extract the
    relevant slice and subtract clip_start so the ASS timecodes are clip-relative."""
    with open(full_transcript_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    result_segments = []
    for seg in data.get("segments", []):
        seg_s = seg.get("start", 0)
        seg_e = seg.get("end", 0)
        # Keep segment if it overlaps the clip window (even partially)
        if seg_e <= clip_start or seg_s >= clip_end:
            continue
        # Clip-local timestamps (clamped to [0, duration])
        local_s = max(0.0, seg_s - clip_start)
        local_e = min(clip_end - clip_start, seg_e - clip_start)
        words_in = []
        for w in seg.get("words", []):
            ws, we = w.get("start", seg_s), w.get("end", seg_e)
            if we <= clip_start or ws >= clip_end:
                continue
            words_in.append({
                "word": w.get("word", ""),
                "start": max(0.0, ws - clip_start),
                "end":   min(clip_end - clip_start, we - clip_start),
            })
        result_segments.append({
            "start": round(local_s, 3),
            "end":   round(local_e, 3),
            "text":  seg.get("text", "").strip(),
            "words": words_in,
        })
    return {"segments": result_segments}


def transcribe_clip(audio_path: str, job_dir: str, clip_index: int, log: DiagnosticLog,
                    clip_start: float = None, clip_end: float = None,
                    full_transcript_path: str = None, translate: bool = True) -> dict:
    """Get word-level timestamps for one clip.

    FAST PATH: if full_transcript_path (or a discoverable one in job_dir) exists
    and clip_start/clip_end are supplied, slices the JSON — zero API calls.
    SLOW PATH: falls back to a direct Deepgram nova-3 call on clip audio.

    translate=True translates here (per-clip, used by the standalone slow path).
    translate=False skips translation so a caller can BATCH all clips in one call."""
    # Resolve the best available full-video transcript
    if full_transcript_path is None or not os.path.exists(full_transcript_path):
        deepgram_full = os.path.join(job_dir, "transcript_deepgram.json")
        whisper_full  = os.path.join(job_dir, "transcript_full.json")
        full_transcript_path = deepgram_full if os.path.exists(deepgram_full) else (
                               whisper_full  if os.path.exists(whisper_full)  else None)

    if clip_start is not None and clip_end is not None and full_transcript_path:
        log.log(f"[Clip {clip_index}] Slicing full-video transcript [{clip_start:.2f}-{clip_end:.2f}s] "
                f"(no Deepgram call)")
        data = _slice_transcript(full_transcript_path, clip_start, clip_end)
    else:
        # ── Slow path: fresh Deepgram call ──
        log.log(f"[Clip {clip_index}] Transcribing clip audio with Deepgram nova-3 "
                f"({'no full transcript found' if full_transcript_path is None else 'no timestamps given'})...")
        data = _deepgram_transcribe(audio_path, log)

    segments = data.get("segments", [])
    words_total = sum(len(s.get("words", [])) for s in segments)
    log.log(f"  Clip segments : {len(segments)}  |  Clip words: {words_total}")

    if segments and translate:
        _translate_segments_to_english(segments, log=log)

    # JSON (machine-readable, full detail)
    transcript_path = os.path.join(job_dir, f"transcript_clip_{clip_index}.json")
    with open(transcript_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

    transcripts_txt = os.path.join(job_dir, "clip_transcripts.txt")
    with open(transcripts_txt, "a", encoding="utf-8") as tf:
        tf.write("=" * 70 + "\n")
        tf.write(f"CLIP {clip_index}  ({len(segments)} segments, {words_total} words)\n")
        tf.write("=" * 70 + "\n")
        for seg in segments:
            tf.write(f"[{seg['start']:7.2f} - {seg['end']:7.2f}]\n")
            tf.write(f"   HI: {seg.get('text', '').strip()}\n")
            tf.write(f"   EN: {seg.get('text_en', '').strip()}\n")
        tf.write("\n")

    log.log(f"  Clip transcript JSON saved: {transcript_path}")
    log.log(f"  Clip transcript text appended: {transcripts_txt}")
    return data


# ─────────────────────────────────────────────────────────────
# Translation (Hindi/Devanagari -> English) for the bottom subtitle track
# ─────────────────────────────────────────────────────────────

def _translate_segments_to_english(segments: list, log: DiagnosticLog):
    """Translate each segment's Devanagari text into natural English and store it
    on seg['text_en']. The original Devanagari seg['text'] / per-word timings are
    left untouched (those drive the Hindi top track). Mutates segments in place."""
    log.section("TRANSLATION (Hindi -> English)")

    # Safe default so the bottom track is never empty if translation is unavailable.
    for seg in segments:
        seg.setdefault("text_en", "")

    texts = [seg["text"].strip() for seg in segments]
    results = _translate_texts(texts, log)
    for i, seg in enumerate(segments):
        if i < len(results) and results[i]:
            seg["text_en"] = results[i]


def _translate_texts(texts: list, log: DiagnosticLog) -> list:
    """Translate a flat list of Hindi strings to English in ONE Groq call.
    Returns a list of English strings aligned by index (missing -> '')."""
    out = [""] * len(texts)
    if not texts:
        return out

    bulk_text = "\n".join(f"{i}:: {t}" for i, t in enumerate(texts))

    prompt = """You are an expert Hindi-to-English translator for video subtitles.
Translate each numbered line of Hindi (Devanagari) into natural, conversational English.
CRITICAL RULES:
1. Keep the EXACT line numbers and prefix format (e.g., 0:: ).
2. Output exactly one line per input line, in the same order.
3. Translate MEANING into fluent English - do NOT transliterate, do NOT keep Hindi words.
4. Keep each translation concise enough to read as a subtitle (it must fit on screen).
TEXT:\n""" + bulk_text

    raw_content = providers.chat(prompt, temperature=0.2, log=log)
    if not raw_content:
        log.log("  Translation unavailable (all providers failed) — English track will be blank")
        return out
    raw_content = re.sub(r"```[a-zA-Z]*\n", "", raw_content).replace("`" * 3, "")
    for line in raw_content.split("\n"):
        if not line.strip() or "::" not in line:
            continue
        try:
            parts = line.split("::", 1)
            idx_match = re.search(r"\d+", parts[0].strip())
            if not idx_match:
                continue
            idx = int(idx_match.group())
            if 0 <= idx < len(out):
                out[idx] = parts[1].strip()
        except Exception:
            continue
    return out


def batch_translate_clips(clips_segments: list, log: DiagnosticLog) -> None:
    """STAGE 4 — translate EVERY segment of EVERY clip in a SINGLE Groq call.

    clips_segments: list of per-clip segment lists (each seg has 'text', gets 'text_en').
    Flattens all segments across all clips, translates once, writes results back.
    This replaces N per-clip translation calls with exactly 1."""
    log.section("BATCH TRANSLATION (all clips, one call)")
    flat = []          # (clip_i, seg_i)
    texts = []
    for ci, segs in enumerate(clips_segments):
        for si, seg in enumerate(segs):
            seg.setdefault("text_en", "")
            flat.append((ci, si))
            texts.append(seg["text"].strip())

    if not texts:
        log.log("  No segments to translate.")
        return

    log.log(f"  Translating {len(texts)} segments from {len(clips_segments)} clips in one call...")
    results = _translate_texts(texts, log)
    for (ci, si), en in zip(flat, results):
        if en:
            clips_segments[ci][si]["text_en"] = en
    log.log("  Batch translation complete.")


# ─────────────────────────────────────────────────────────────
# Transliteration (Hindi/Devanagari -> Roman "Hinglish") for a Latin-script track
# ─────────────────────────────────────────────────────────────

def _transliterate_texts(texts: list, log: DiagnosticLog) -> list:
    """Romanise a flat list of Hindi (Devanagari) strings into natural Hinglish in
    ONE Groq call. Hinglish = Hindi words written in Latin/Roman letters the way
    Indians type online (common English words kept as English).
    Returns a list aligned by index (missing -> '')."""
    out = [""] * len(texts)
    if not texts:
        return out

    bulk_text = "\n".join(f"{i}:: {t}" for i, t in enumerate(texts))

    prompt = """You are an expert at writing Hinglish subtitles.
For each numbered line of Hindi (Devanagari), write the SAME sentence in ROMAN/LATIN letters (Hinglish)
- the way Indians casually type Hindi in English script.
CRITICAL RULES:
1. Keep the EXACT line numbers and prefix format (e.g., 0:: ).
2. Output exactly one line per input line, in the same order.
3. Do NOT translate the meaning. ROMANISE the Hindi sounds (e.g. "मैं ठीक हूँ" -> "main theek hoon").
4. Keep common English words that appear as English. Use everyday, readable spelling (no accents/diacritics).
TEXT:\n""" + bulk_text

    raw_content = providers.chat(prompt, temperature=0.2, log=log)
    if not raw_content:
        log.log("  Transliteration unavailable (all providers failed) — Hinglish track will be blank")
        return out
    raw_content = re.sub(r"```[a-zA-Z]*\n", "", raw_content).replace("`" * 3, "")
    for line in raw_content.split("\n"):
        if not line.strip() or "::" not in line:
            continue
        try:
            parts = line.split("::", 1)
            idx_match = re.search(r"\d+", parts[0].strip())
            if not idx_match:
                continue
            idx = int(idx_match.group())
            if 0 <= idx < len(out):
                out[idx] = parts[1].strip()
        except Exception:
            continue
    return out


def batch_transliterate_clips(clips_segments: list, log: DiagnosticLog) -> None:
    """STAGE 4 — romanise EVERY segment of EVERY clip into Hinglish in a SINGLE Groq
    call and write the result onto seg['text_hinglish']. Mirrors batch_translate_clips."""
    log.section("BATCH TRANSLITERATION (Hindi -> Hinglish, one call)")
    flat = []
    texts = []
    for ci, segs in enumerate(clips_segments):
        for si, seg in enumerate(segs):
            seg.setdefault("text_hinglish", "")
            flat.append((ci, si))
            texts.append(seg["text"].strip())

    if not texts:
        log.log("  No segments to transliterate.")
        return

    log.log(f"  Transliterating {len(texts)} segments from {len(clips_segments)} clips in one call...")
    results = _transliterate_texts(texts, log)
    for (ci, si), hg in zip(flat, results):
        if hg:
            clips_segments[ci][si]["text_hinglish"] = hg
    log.log("  Batch transliteration complete.")


def batch_generate_titles(clips_segments: list, log: DiagnosticLog) -> list:
    """STAGE 4 — generate ONE crisp on-screen title per clip in a SINGLE Groq call.

    Returns a list of title strings aligned to clips_segments order ('' if none).
    The title is short (<= 6 words), punchy, English, and based on the clip's content."""
    log.section("BATCH TITLE GENERATION (all clips, one call)")
    titles = ["" for _ in clips_segments]

    # Build a compact prompt: one numbered block of Hindi text per clip.
    blocks = []
    for ci, segs in enumerate(clips_segments):
        hi = " ".join((s.get("text") or "").strip() for s in segs).strip()
        if not hi:
            hi = "(no transcript)"
        blocks.append(f"{ci}:: {hi[:600]}")   # cap length per clip to keep prompt small
    bulk = "\n".join(blocks)

    prompt = """You are a viral short-form video editor. For each numbered clip below, write ONE
punchy on-screen TITLE that would make someone stop scrolling. Rules:
1. Keep the EXACT line numbers and prefix (e.g., 0:: ).
2. Max 6 words. No quotes, no emojis, no hashtags, no ending punctuation.
3. English. Make it a curiosity hook or bold statement tied to the clip's content.
4. Output exactly one line per clip, same order.
CLIPS:\n""" + bulk

    raw = providers.chat(prompt, temperature=0.6, log=log)
    if not raw:
        log.log("  Title generation unavailable (all providers failed) — no titles")
        return titles
    raw = re.sub(r"```[a-zA-Z]*\n", "", raw).replace("`" * 3, "")
    for line in raw.split("\n"):
        if "::" not in line:
            continue
        try:
            pfx, val = line.split("::", 1)
            m = re.search(r"\d+", pfx)
            if not m:
                continue
            idx = int(m.group())
            if 0 <= idx < len(titles):
                titles[idx] = val.strip().strip('"').strip()
        except Exception:
            continue
    log.log(f"  Generated {sum(1 for t in titles if t)}/{len(titles)} titles.")
    return titles


# ─────────────────────────────────────────────────────────────
# SRT helpers (built from a CLIP-LOCAL transcript, no time remap needed)
# ─────────────────────────────────────────────────────────────

def _fmt_srt_time(seconds: float) -> str:
    frac = int((seconds % 1) * 1000)
    s = int(seconds)
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d},{frac:03d}"

def _fmt_ass_time(seconds: float) -> str:
    if seconds < 0:
        seconds = 0.0
    cs = int(round((seconds % 1) * 100))
    s = int(seconds)
    if cs == 100:
        cs = 0
        s += 1
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    return f"{h:d}:{m:02d}:{s:02d}.{cs:02d}"


def _ass_escape(text: str) -> str:
    return (text or "").replace("\\", "\\\\").replace("{", "(").replace("}", ")").replace("\n", " ").strip()


# Subtitle chunking target: 5-10 words per on-screen frame.
_WORDS_PER_CUE_MIN = 2
_WORDS_PER_CUE_MAX = 3


def _chunk_word_cues(segments: list, lead_offset: float, clip_duration,
                     max_words: int = _WORDS_PER_CUE_MAX) -> list:
    """Group Devanagari words (with real timestamps) into cues of up to max_words,
    preferring to break at the end of a segment (natural pause). Each cue is timed
    from its first word's start to its last word's end. Returns (start, end, text).

    Fallback: when a segment has no word-level timestamps but has text, the segment
    duration is divided evenly across its words so subtitles still appear."""
    cues = []
    bucket = []  # list of (start, end, word)
    for seg in segments:
        seg_words = [w for w in seg.get("words", [])
                     if (w.get("word") or "").strip() and w.get("start") is not None and w.get("end") is not None]

        # Fallback: no word timestamps — synthesise them from the segment span.
        if not seg_words:
            text = (seg.get("text") or "").strip()
            if not text:
                continue
            s0 = max(0.0, float(seg.get("start", 0)) - lead_offset)
            e0 = max(0.0, float(seg.get("end", 0)) - lead_offset)
            if clip_duration is not None:
                e0 = min(e0, clip_duration)
            toks = text.split()
            if toks and e0 > s0:
                span = (e0 - s0) / len(toks)
                seg_words = [(s0 + i * span, s0 + (i + 1) * span, t) for i, t in enumerate(toks)]
            else:
                continue

        for item in seg_words:
            if isinstance(item, tuple):
                ws, we, word = item
            else:
                ws = max(0.0, item["start"] - lead_offset)
                we = max(0.0, item["end"] - lead_offset)
                word = (item["word"] or "").strip()
            bucket.append((ws, we, word))
            if len(bucket) >= max_words:
                cues.append(_flush_bucket(bucket, clip_duration)); bucket = []
        # Prefer a break at the segment boundary once we have a readable amount.
        if len(bucket) >= _WORDS_PER_CUE_MIN:
            cues.append(_flush_bucket(bucket, clip_duration)); bucket = []
    if bucket:
        cues.append(_flush_bucket(bucket, clip_duration))
    # Enforce no-overlap: each cue ends no later than the next cue starts.
    valid = [c for c in cues if c]
    for i in range(len(valid) - 1):
        s, e, t = valid[i]
        next_s = valid[i + 1][0]
        if e > next_s:
            valid[i] = (s, next_s, t)
    return valid


def _flush_bucket(bucket, clip_duration):
    if not bucket:
        return None
    c_start, c_end = bucket[0][0], bucket[-1][1]
    if clip_duration is not None:
        c_end = min(c_end, clip_duration)
    if c_end <= c_start:
        c_end = c_start + 0.3
    return (c_start, c_end, " ".join(b[2] for b in bucket))


def _chunk_text_cues(segments: list, text_key: str, lead_offset: float, clip_duration,
                     max_words: int = _WORDS_PER_CUE_MAX) -> list:
    """For text without per-word timings (the English translation), split each
    segment's text into <=max_words chunks and distribute the segment's time span
    evenly across them. Returns (start, end, text)."""
    cues = []
    for seg in segments:
        text = (seg.get(text_key) or "").strip()
        if not text:
            continue
        seg_s = max(0.0, float(seg["start"]) - lead_offset)
        seg_e = max(0.0, float(seg["end"]) - lead_offset)
        if clip_duration is not None:
            seg_e = min(seg_e, clip_duration)
        if seg_e <= seg_s:
            continue
        words = text.split()
        # number of chunks needed for this segment
        n_chunks = max(1, math.ceil(len(words) / max_words))
        per = math.ceil(len(words) / n_chunks)
        span = (seg_e - seg_s) / n_chunks
        for i in range(n_chunks):
            piece = " ".join(words[i * per:(i + 1) * per]).strip()
            if not piece:
                continue
            cs = seg_s + i * span
            ce = seg_s + (i + 1) * span
            cues.append((cs, ce, piece))
    return cues


# Visual tuning (real pixels on the 1080x1920 frame)
_HI_FONTSIZE    = 60
_TITLE_FONTSIZE = 64
_EN_FONTSIZE    = 54
_SINGLE_FONTSIZE = 58
_SIDE_MARGIN    = 70
_OUTLINE        = 5
_TITLE_COLOUR   = "&H0033E6FF"   # warm yellow (ASS = AABBGGRR) for the crisp title

# Single-track caption positions:
#   top    -> above the video band (in the upper letterbox bar)
#   middle -> centred over the video
#   bottom -> ON the video, near its bottom edge (overlaid on the footage)
#   below  -> just BELOW the video band (in the lower letterbox bar, outside the footage)
# Values below are the fallback (no known video geometry).
_POS_MARGIN_V = {"top": 120, "middle": 0, "bottom": 170, "below": 170}
_POS_ALIGN    = {"top": 8, "middle": 5, "bottom": 2, "below": 2}

# When the source isn't already 9:16 it gets letterboxed (black bars top/bottom).
# "below"/"top" sit in those bars (just outside the footage); "bottom" sits ON the
# footage a little above its bottom edge.
_LETTERBOX_GAP = 26          # px gap between the video band and an OUTSIDE caption
_ON_VIDEO_INSET = 70         # px above the video's bottom edge for an ON-video caption
_MIN_BAR_FOR_OUTSIDE = 170   # need at least this much bar to seat a caption outside the video


def _video_box(src_w, src_h):
    """Given the SOURCE w/h, return (video_top_y, video_bottom_y) of the scaled video
    band inside the 1080x1920 frame (matching the scale+pad render filter). None if
    dimensions are unknown."""
    if not src_w or not src_h:
        return None
    scale = min(SHORTS_W / float(src_w), SHORTS_H / float(src_h))
    vh = src_h * scale
    pad_top = (SHORTS_H - vh) / 2.0
    return (pad_top, pad_top + vh)


def _position_layout(position, video_box):
    """Return (ass_alignment, margin_v) for a single caption.
      top    -> just ABOVE the video band (upper bar) when there's room
      middle -> centred on the video
      bottom -> ON the video, a little above its bottom edge
      below  -> just BELOW the video band (lower bar) when there's room
    Falls back to in-frame margins when geometry is unknown or there's no bar."""
    if position not in _POS_ALIGN:
        position = "bottom"
    if not video_box:
        return _POS_ALIGN[position], _POS_MARGIN_V[position]
    v_top, v_bot = video_box
    lower_bar = SHORTS_H - v_bot
    upper_bar = v_top
    if position == "middle":
        return 5, 0
    if position == "bottom":
        # bottom-anchored, sitting INSIDE the footage just above its bottom edge
        return 2, max(0, int(round(lower_bar + _ON_VIDEO_INSET)))
    if position == "below":
        if lower_bar >= _MIN_BAR_FOR_OUTSIDE:
            return 8, int(round(v_bot + _LETTERBOX_GAP))   # top-anchored, just under the video
        return 2, _POS_MARGIN_V["bottom"]                  # no bar -> fall back onto the video
    if position == "top":
        if upper_bar >= _MIN_BAR_FOR_OUTSIDE:
            return 2, int(round(SHORTS_H - v_top + _LETTERBOX_GAP))  # bottom-anchored, just above video
        return 8, _POS_MARGIN_V["top"]
    return _POS_ALIGN[position], _POS_MARGIN_V[position]


def _probe_dimensions(path, log):
    """Return (width, height) of the first video stream, or (None, None)."""
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height", "-of", "csv=p=0", path],
            capture_output=True, text=True)
        parts = r.stdout.strip().split(",")
        return int(parts[0]), int(parts[1])
    except Exception:
        return None, None

# Colours (ASS = AABBGGRR, where AA=00 is fully opaque)
_WHITE = "&H00FFFFFF"
_BLACK = "&H00000000"
_SHADOW_BACK = "&H64000000"   # translucent black drop shadow (outline style)
_BOX_BACK    = "&HA0000000"   # mostly-opaque black box (box style)
_WHITE_BOX_BACK = "&H80FFFFFF"  # 50% white / 50% transparent slab (white_box style; dark text)
_ORANGE      = "&H0000A5FF"   # RGB FF A5 00 — vibrant orange (fire)
_DARK_RED    = "&H000000CC"   # RGB CC 00 00 — deep red (fire outline)
_MAGENTA_BOX = "&HA0FF00FF"   # 63% opaque magenta slab (retro box)

# One reusable Style-row tail. Order matches the Format line below.
_STYLE_FIELDS = "1,0,0,0,100,100,0,0,{bs},{ol},{sh},{al},{ml},{mr},{mv},1"

# A few punchy accent colours (ASS = AABBGGRR).
_YELLOW = "&H0000FFFF"   # RGB FFFF00
_GREEN  = "&H0040E62E"   # RGB 2EE640 (vivid lime)
_PINK   = "&H00B469FF"   # RGB FF69B4 (hot pink)
_CYAN   = "&H00FFFF00"   # RGB 00FFFF
_DEFAULT_ACCENT = _YELLOW


def _hex_to_ass(hexstr: str, default: str = _DEFAULT_ACCENT) -> str:
    """Convert '#RRGGBB' (web hex) into an ASS '&H00BBGGRR' colour string."""
    if not hexstr:
        return default
    s = str(hexstr).strip().lstrip("#")
    if len(s) != 6:
        return default
    try:
        r, g, b = int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16)
    except ValueError:
        return default
    return f"&H00{b:02X}{g:02X}{r:02X}"


# Trendy caption presets. Each bundles size, colours, border + an optional animation.
#   anim: None      -> static text
#         "karaoke" -> phrase stays on screen, the spoken word lights up (Hormozi look)
#         "fade"    -> each cue fades in/out smoothly
# Recognised names: outline, box, white_box, bold_yellow, karaoke,
#                   neon, retro, shadow, fire, fade.
def _style_preset(name: str, accent: str = _DEFAULT_ACCENT) -> dict:
    name = (name or "outline").lower()
    p = dict(fontsize=_SINGLE_FONTSIZE, primary=_WHITE, accent=accent,
             outline=_OUTLINE, border=1, shadow=1, back=_SHADOW_BACK,
             outline_colour=_BLACK, upper=False, anim=None)
    if name == "box":
        p.update(border=3, outline=6, shadow=0, back=_BOX_BACK)
    elif name == "white_box":
        # Dark text on a solid whitish slab (BorderStyle 3: outline colour = box padding).
        p.update(primary=_BLACK, border=3, outline=8, shadow=0,
                 back=_WHITE_BOX_BACK, outline_colour=_WHITE_BOX_BACK)
    elif name == "bold_yellow":
        p.update(fontsize=66, primary=_YELLOW, outline=6, upper=True)
    elif name == "karaoke":
        p.update(fontsize=62, outline=5, upper=True, anim="karaoke")
    elif name == "neon":
        # Cyan text, thick white stroke — glowing neon look, uppercase
        p.update(fontsize=64, primary=_CYAN, outline_colour=_WHITE, outline=4, shadow=3, upper=True, back=_BLACK)
    elif name == "retro":
        # White text on a magenta/pink opaque slab — TikTok retro aesthetic, uppercase
        p.update(fontsize=60, primary=_WHITE, border=3, outline=8, shadow=0,
                 back=_MAGENTA_BOX, outline_colour=_MAGENTA_BOX, upper=True)
    elif name == "shadow":
        # Cinematic: white text, large drop shadow, minimal outline (Netflix/film look)
        p.update(fontsize=62, primary=_WHITE, outline=1, shadow=7, border=1, back="&HA0000000")
    elif name == "fire":
        # Orange text, dark-red thick outline, uppercase — high energy
        p.update(fontsize=66, primary=_ORANGE, outline_colour=_DARK_RED, outline=6, shadow=2, upper=True, back=_BLACK)
    elif name == "fade":
        # Standard outline but each cue fades in and out smoothly
        p.update(fontsize=60, outline=5, anim="fade")
    # "outline" == the default base
    return p


# Styles that are static text (valid for the dual layout); animated ones fall back
# to "outline" in dual since two animated tracks at once is visual noise.
_STATIC_STYLES = ("outline", "box", "white_box", "bold_yellow", "neon", "retro", "shadow", "fire", "fade")
VALID_CAPTION_STYLES = ("outline", "box", "white_box", "bold_yellow", "karaoke", "neon", "retro", "shadow", "fire", "fade")


def _style_row(name: str, font: str, preset: dict, align: int, margin_v: int,
               primary: str = None, fontsize: int = None) -> str:
    """Build one ASS 'Style:' line from a preset (with optional primary/size override)."""
    primary = primary or preset["primary"]
    size = fontsize or preset["fontsize"]
    oc = preset.get("outline_colour", _BLACK)
    tail = _STYLE_FIELDS.format(bs=preset["border"], ol=preset["outline"], sh=preset["shadow"],
                                al=align, ml=_SIDE_MARGIN, mr=_SIDE_MARGIN, mv=margin_v)
    return f"Style: {name},{font},{size},{primary},{primary},{oc},{preset['back']},{tail}"


def _text_key_for(language: str) -> str:
    return {"english": "text_en", "hinglish": "text_hinglish"}.get(language, "text")


def _esc_word(text: str) -> str:
    """Escape a single word for an ASS Dialogue Text (no brace->paren swap needed for
    plain words, but stay safe in case punctuation sneaks in)."""
    return (text or "").replace("\\", "\\\\").replace("{", "(").replace("}", ")").strip()


def _word_timings(seg: dict, text_key: str, lead_offset: float, clip_duration) -> list:
    """Return [(start, end, word)] for a segment. Uses REAL word timestamps when the
    Devanagari source has them; otherwise splits the chosen text evenly across the
    segment span (so animated styles still work for English/Hinglish)."""
    words_meta = [w for w in seg.get("words", [])
                  if (w.get("word") or "").strip() and w.get("start") is not None and w.get("end") is not None]
    out = []
    if words_meta and text_key == "text":
        for w in words_meta:
            s = max(0.0, w["start"] - lead_offset)
            e = max(0.0, w["end"] - lead_offset)
            if clip_duration is not None:
                e = min(e, clip_duration)
            if e > s:
                out.append((s, e, (w["word"] or "").strip()))
        return out
    text = (seg.get(text_key) or "").strip()
    if not text:
        return out
    s0 = max(0.0, float(seg["start"]) - lead_offset)
    e0 = max(0.0, float(seg["end"]) - lead_offset)
    if clip_duration is not None:
        e0 = min(e0, clip_duration)
    toks = text.split()
    if not toks or e0 <= s0:
        return out
    span = (e0 - s0) / len(toks)
    for i, t in enumerate(toks):
        out.append((s0 + i * span, s0 + (i + 1) * span, t))
    return out


def _all_word_timings(segments: list, text_key: str, lead_offset: float, clip_duration) -> list:
    words = []
    for seg in segments:
        words.extend(_word_timings(seg, text_key, lead_offset, clip_duration))
    return words


def _karaoke_events(segments, text_key, lead_offset, clip_duration, preset, group=3) -> list:
    """Hormozi-style: a short phrase stays on screen and the currently spoken word is
    recoloured + slightly enlarged. Returns [(start, end, ass_text)]."""
    words = _all_word_timings(segments, text_key, lead_offset, clip_duration)
    accent = preset["accent"]
    events = []
    for i in range(0, len(words), group):
        chunk = words[i:i + group]
        disp = [(w[2].upper() if preset["upper"] else w[2]) for w in chunk]
        for j, (s, e, _w) in enumerate(chunk):
            parts = []
            for k, word in enumerate(disp):
                wesc = _esc_word(word)
                if k == j:
                    parts.append(f"{{\\1c{accent}\\fscx116\\fscy116}}{wesc}{{\\r}}")
                else:
                    parts.append(wesc)
            events.append((s, e, " ".join(parts)))
    return events


def _wordpop_events(segments, text_key, lead_offset, clip_duration, preset) -> list:
    """Fast-cut Reels style: ONE word on screen at a time, scaling/fading in."""
    words = _all_word_timings(segments, text_key, lead_offset, clip_duration)
    events = []
    for s, e, w in words:
        word = w.upper() if preset["upper"] else w
        text = (f"{{\\fad(50,40)\\fscx72\\fscy72\\t(0,120,\\fscx100\\fscy100)}}"
                f"{_esc_word(word)}")
        events.append((s, e, text))
    return events


def _static_cues(segments: list, language: str, lead_offset: float, clip_duration):
    """Plain chunked cues (no animation) for the chosen language."""
    if language == "english":
        return _chunk_text_cues(segments, "text_en", lead_offset, clip_duration)
    if language == "hinglish":
        return _chunk_text_cues(segments, "text_hinglish", lead_offset, clip_duration)
    return _chunk_word_cues(segments, lead_offset, clip_duration)


def make_caption_ass(segments: list, ass_path: str,
                     layout: str = "single",
                     language: str = "hindi",
                     position: str = "bottom",
                     caption_style: str = "outline",
                     accent_color: str = "",
                     title: str = "",
                     show_title: bool = False,
                     hindi_font: str = "Noto Sans Devanagari",
                     latin_font: str = "Poppins",
                     lead_offset: float = 0.08,
                     clip_duration: float = None,
                     video_box: tuple = None) -> tuple:
    """Write ONE ASS file on a 1080x1920 frame.

    layout == "single": ONE caption track in `language`, pinned to `position`, styled
                        by `caption_style` (outline / box / bold_yellow / karaoke /
                        word_pop). Animated styles (karaoke, word_pop) light up or pop
                        word-by-word; `accent_color` ('#RRGGBB') tints the active word.
    layout == "dual"  : the classic two-track look — Devanagari Hindi on TOP and the
                        English translation on the BOTTOM (uses a static style).

    show_title adds a static AI headline (placed opposite the captions). Returns
    (primary_cue_count, secondary_cue_count, has_title).
    """
    position = position if position in _POS_ALIGN else "bottom"
    title = (title or "").strip()
    accent = _hex_to_ass(accent_color) if accent_color else _DEFAULT_ACCENT
    styles = []
    events = []

    # Title always uses a crisp static headline style.
    title_preset = _style_preset("bold_yellow", accent)

    def add_title(default_pos: str):
        if not (show_title and title):
            return False
        align, mv = _position_layout(default_pos, video_box)
        styles.append(_style_row("TITLE", latin_font, title_preset, align, mv,
                                  primary=_TITLE_COLOUR, fontsize=_TITLE_FONTSIZE))
        end = clip_duration if clip_duration else 3600
        events.append(f"Dialogue: 0,{_fmt_ass_time(0)},{_fmt_ass_time(end)},TITLE,,0,0,0,,"
                      f"{_ass_escape(title.upper())}")
        return True

    if layout == "dual":
        # Classic two-track look — animated styles don't apply here, fall back to static.
        dual_name = caption_style if caption_style in _STATIC_STYLES else "outline"
        dpreset = _style_preset(dual_name, accent)
        hi_cues = _chunk_word_cues(segments, lead_offset, clip_duration)
        en_cues = _chunk_text_cues(segments, "text_en", lead_offset, clip_duration)
        styles.append(_style_row("HI", hindi_font, dpreset, 8, 90, fontsize=_HI_FONTSIZE))
        styles.append(_style_row("EN", latin_font, dpreset, 2, 150, fontsize=_EN_FONTSIZE))
        has_title = False
        if show_title and title:
            styles.append(_style_row("TITLE", latin_font, title_preset, 8, 270,
                                      primary=_TITLE_COLOUR, fontsize=_TITLE_FONTSIZE))
            end = clip_duration if clip_duration else 3600
            events.append(f"Dialogue: 0,{_fmt_ass_time(0)},{_fmt_ass_time(end)},TITLE,,0,0,0,,"
                          f"{_ass_escape(title.upper())}")
            has_title = True
        up = dpreset["upper"]
        for c_start, c_end, text in hi_cues:
            events.append(f"Dialogue: 0,{_fmt_ass_time(c_start)},{_fmt_ass_time(c_end)},HI,,0,0,0,,{_ass_escape(text)}")
        for c_start, c_end, text in en_cues:
            t = text.upper() if up else text
            events.append(f"Dialogue: 0,{_fmt_ass_time(c_start)},{_fmt_ass_time(c_end)},EN,,0,0,0,,{_ass_escape(t)}")
        primary_count, secondary_count = len(hi_cues), len(en_cues)
    else:
        # Single track: one chosen language at one position, in the chosen style.
        preset = _style_preset(caption_style, accent)
        text_key = _text_key_for(language)
        font = hindi_font if language == "hindi" else latin_font
        sub_align, sub_mv = _position_layout(position, video_box)
        styles.append(_style_row("SUB", font, preset, sub_align, sub_mv))

        if preset["anim"] == "karaoke":
            ev = _karaoke_events(segments, text_key, lead_offset, clip_duration, preset)
        elif preset["anim"] == "fade":
            cues = _static_cues(segments, language, lead_offset, clip_duration)
            ev = [(s, e, f'{{\\fad(120,80)}}{_ass_escape(t.upper() if preset["upper"] else t)}')
                  for s, e, t in cues]
        else:
            cues = _static_cues(segments, language, lead_offset, clip_duration)
            ev = [(s, e, _ass_escape(t.upper() if preset["upper"] else t)) for s, e, t in cues]

        # Title goes opposite the captions to avoid overlap.
        title_pos = "below" if position == "top" else "top"
        has_title = add_title(title_pos)
        for s, e, text in ev:
            events.append(f"Dialogue: 0,{_fmt_ass_time(s)},{_fmt_ass_time(e)},SUB,,0,0,0,,{text}")
        primary_count, secondary_count = len(ev), 0

    header = (
        "[Script Info]\n"
        "ScriptType: v4.00+\n"
        f"PlayResX: {SHORTS_W}\nPlayResY: {SHORTS_H}\n"
        "WrapStyle: 0\nScaledBorderAndShadow: yes\n\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, "
        "BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, "
        "BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n"
        + "\n".join(styles) + "\n\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )

    with open(ass_path, "w", encoding="utf-8") as f:
        f.write(header)
        f.write("\n".join(events))
        f.write("\n")

    return primary_count, secondary_count, has_title


def _get_media_duration(path: str, log: DiagnosticLog) -> float:
    """Returns duration in seconds via ffprobe, or None on failure."""
    command = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        path,
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        log.log(f"     ffprobe failed for {path}: {result.stderr.strip()}")
        return None
    try:
        return float(result.stdout.strip())
    except ValueError:
        return None


# ─────────────────────────────────────────────────────────────
# FFmpeg render filter builder (9:16 Shorts canvas + dual subtitles)
# ─────────────────────────────────────────────────────────────

# YouTube Shorts canvas
SHORTS_W, SHORTS_H = 1080, 1920

def _escape_ffmpeg_path(path: str) -> str:
    path = path.replace("\\", "/")
    path = path.replace("'", "\\'")
    path = path.replace(":", "\\:")
    return path


def _build_render_filter(ass_path: str, fontsdir: str = "") -> str:
    """Full -vf chain for a YouTube Short:
       1. scale the source to fit a 1080x1920 frame WITHOUT cropping, centered
          (black bars top/bottom leave room for the two caption tracks)
       2. burn the dual-style ASS (Hindi on top, English on bottom)
    `fontsdir` should point at the bundled Devanagari font folder; the Latin font
    (Poppins) is resolved from system fonts via fontconfig.
    """
    # bilinear downscaling is noticeably cheaper than the default bicubic with
    # negligible quality loss at this resolution; override with SCALE_FLAGS if needed.
    scale_flags = os.environ.get("SCALE_FLAGS", "bilinear")
    chain = [
        f"scale={SHORTS_W}:{SHORTS_H}:force_original_aspect_ratio=decrease:flags={scale_flags}",
        f"pad={SHORTS_W}:{SHORTS_H}:(ow-iw)/2:(oh-ih)/2:color=black",
        "setsar=1",
    ]
    if ass_path:
        ass_esc = _escape_ffmpeg_path(ass_path)
        if fontsdir and os.path.isdir(fontsdir):
            chain.append(f"subtitles='{ass_esc}':fontsdir='{_escape_ffmpeg_path(fontsdir)}'")
        else:
            chain.append(f"subtitles='{ass_esc}'")
    return ",".join(chain)


# ─────────────────────────────────────────────────────────────
# Fonts
# ─────────────────────────────────────────────────────────────
# IMPORTANT: For Devanagari (Hindi) captions to render, a Devanagari-capable font
# MUST be reachable. We search the bundled fonts/ folder first, then several common
# system locations. If NONE is found, Hindi captions fall back to the renderer's
# default font and may show as empty boxes — the logs will warn loudly in that case.
# (This is the bug that caused "subtitles didn't load" when the font file was missing.)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# (file_path, font_family_name) — family name is what the ASS style references.
_HINDI_FONT_CANDIDATES = [
    (os.path.join(BASE_DIR, "fonts", "NotoSansDevanagari", "full", "ttf", "NotoSansDevanagari-Bold.ttf"), "Noto Sans Devanagari"),
    (os.path.join(BASE_DIR, "fonts", "NotoSansDevanagari-Bold.ttf"), "Noto Sans Devanagari"),
    (os.path.join(BASE_DIR, "fonts", "NotoSansDevanagari-Regular.ttf"), "Noto Sans Devanagari"),
    (os.path.join(BASE_DIR, "fonts", "Baloo2-Bold.ttf"), "Baloo 2"),
    (os.path.join(BASE_DIR, "fonts", "Laila-Bold.ttf"), "Laila"),
    (os.path.join(BASE_DIR, "fonts", "Rajdhani-Bold.ttf"), "Rajdhani"),
    ("/usr/share/fonts/truetype/noto/NotoSansDevanagari-Bold.ttf", "Noto Sans Devanagari"),
    ("/usr/share/fonts/truetype/noto/NotoSansDevanagari-Regular.ttf", "Noto Sans Devanagari"),
    ("/usr/share/fonts/truetype/lohit-devanagari/Lohit-Devanagari.ttf", "Lohit Devanagari"),
    ("/usr/share/fonts/truetype/Sarai/Sarai.ttf", "Sarai"),
]
_LATIN_FONT_CANDIDATES = [
    (os.path.join(BASE_DIR, "fonts", "Poppins-Bold.ttf"), "Poppins"),
    (os.path.join(BASE_DIR, "fonts", "Oswald-Bold.ttf"), "Oswald"),
    (os.path.join(BASE_DIR, "fonts", "Montserrat-Bold.ttf"), "Montserrat"),
    (os.path.join(BASE_DIR, "fonts", "Staatliches-Regular.ttf"), "Staatliches"),
    (os.path.join(BASE_DIR, "fonts", "BarlowCondensed-Bold.ttf"), "Barlow Condensed"),
    (os.path.join(BASE_DIR, "fonts", "Righteous-Regular.ttf"), "Righteous"),
    ("/usr/share/fonts/truetype/google-fonts/Poppins-Bold.ttf", "Poppins"),
    ("/usr/share/fonts/truetype/poppins/Poppins-Bold.ttf", "Poppins"),
    ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "DejaVu Sans"),
    ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "DejaVu Sans"),
]


def _resolve_font(candidates):
    """Return (file_path, family_name) for the first candidate that exists, else ('','')."""
    for path, family in candidates:
        if path and os.path.exists(path):
            return path, family
    return "", ""


def _get_hindi_font():
    """(path, family) for a Devanagari font, or ('','') if none is available."""
    return _resolve_font(_HINDI_FONT_CANDIDATES)


def _get_latin_font():
    """(path, family) for a Latin font, or ('','') if none is available."""
    return _resolve_font(_LATIN_FONT_CANDIDATES)


# ── User-selectable caption fonts ──────────────────────────────────────────
# key -> (family name used in the ASS style, filename expected under ./fonts/).
# Download them with the helper in the README (fetch_fonts script). If a chosen
# font file isn't present, we fall back to whatever Devanagari/Latin font we can find.
HINDI_FONTS = {
    "noto":     ("Noto Sans Devanagari", "NotoSansDevanagari-Bold.ttf"),
    "mukta":    ("Mukta",                "Mukta-Bold.ttf"),
    "hind":     ("Hind",                 "Hind-Bold.ttf"),
    "rozha":    ("Rozha One",            "RozhaOne-Regular.ttf"),
    "kalam":    ("Kalam",                "Kalam-Bold.ttf"),
    "baloo":    ("Baloo 2",              "Baloo2-Bold.ttf"),
    "laila":    ("Laila",                "Laila-Bold.ttf"),
    "rajdhani": ("Rajdhani",             "Rajdhani-Bold.ttf"),
}
ENGLISH_FONTS = {
    "poppins":     ("Poppins",            "Poppins-Bold.ttf"),
    "anton":       ("Anton",              "Anton-Regular.ttf"),
    "bebas":       ("Bebas Neue",         "BebasNeue-Regular.ttf"),
    "archivo":     ("Archivo Black",      "ArchivoBlack-Regular.ttf"),
    "fjalla":      ("Fjalla One",         "FjallaOne-Regular.ttf"),
    "oswald":      ("Oswald",             "Oswald-Bold.ttf"),
    "montserrat":  ("Montserrat",         "Montserrat-Bold.ttf"),
    "staatliches": ("Staatliches",        "Staatliches-Regular.ttf"),
    "barlow":      ("Barlow Condensed",   "BarlowCondensed-Bold.ttf"),
    "righteous":   ("Righteous",          "Righteous-Regular.ttf"),
}
VALID_HINDI_FONTS = tuple(HINDI_FONTS.keys())
VALID_ENGLISH_FONTS = tuple(ENGLISH_FONTS.keys())


def _font_from_choice(choice, table, fallback_resolver):
    """Resolve a user's font choice to (path, family).
    Looks for the chosen font's file under ./fonts/; if it isn't there, falls back
    to the first auto-detected font (so a bad/missing choice never breaks rendering)."""
    entry = table.get((choice or "").strip().lower())
    if entry:
        family, fname = entry
        for cand in (os.path.join(BASE_DIR, "fonts", fname),
                     os.path.join(BASE_DIR, "fonts", family.replace(" ", ""), fname)):
            if os.path.exists(cand):
                return cand, family
    return fallback_resolver()


def _prepare_fontsdir(font_paths, job_dir: str) -> str:
    """ffmpeg's subtitles filter takes a SINGLE fontsdir. If the fonts we need live
    in different folders (e.g. bundled Devanagari + system Latin), copy them into one
    per-job cache dir and return that. Returns '' if no bundled fonts were found."""
    paths = [p for p in font_paths if p and os.path.exists(p)]
    if not paths:
        return ""
    dirs = {os.path.dirname(p) for p in paths}
    if len(dirs) == 1:
        return dirs.pop()
    cache = os.path.join(job_dir, ".fonts")
    os.makedirs(cache, exist_ok=True)
    for p in paths:
        dst = os.path.join(cache, os.path.basename(p))
        if not os.path.exists(dst):
            try:
                shutil.copy2(p, dst)
            except OSError:
                pass
    return cache


# ─────────────────────────────────────────────────────────────
# Subtitle burning for one clip
# ─────────────────────────────────────────────────────────────

def burn_subtitles_for_clip(raw_path: str, clip_index: int, job_dir: str, clips_dir: str,
                             log: DiagnosticLog, clip_callback=None, reason: str = "",
                             clip_start: float = None, clip_end: float = None,
                             segments: list = None, title: str = "",
                             burn: bool = True,
                             layout: str = "single",
                             language: str = "hindi",
                             position: str = "bottom",
                             caption_style: str = "outline",
                             accent_color: str = "",
                             hindi_font_choice: str = "",
                             english_font_choice: str = "",
                             show_title: bool = False,
                             src_dims: tuple = None) -> str:
    """Renders ONE finished 9:16 Short in a single ffmpeg pass.

    `raw_path` is the SOURCE video; we seek into it with -ss/-to instead of cutting a
    separate raw clip first.

    burn=False           -> just cut + scale to clean 9:16, NO captions at all.
    layout="single"      -> one caption track in `language` at `position`.
    layout="dual"        -> Devanagari Hindi top + English bottom (classic look).
    caption_style        -> "outline" (text + outline) or "box" (solid background).
    show_title           -> add the static AI headline overlay.
    """
    final_output = os.path.join(clips_dir, f"viral_clip_{clip_index}.mp4")

    log.log(f"\n   Clip {clip_index}  [{clip_start:.2f}s–{clip_end:.2f}s]"
            if clip_start is not None else f"\n   Clip {clip_index}")
    log.log(f"     Source : {raw_path}")
    log.log(f"     Captions: burn={burn} layout={layout} lang={language} pos={position} "
            f"style={caption_style} title={show_title}")
    log.log(f"     Output : {final_output}")

    if not os.path.exists(raw_path):
        log.log(f"     FAILED - source video does not exist")
        return None

    ass_path = None

    try:
        clip_duration = (clip_end - clip_start) if (clip_start is not None and clip_end is not None) else None
        vf = None

        if not burn:
            # ── No-subtitle path: clean 9:16 video, nothing overlaid. ──
            log.log("     Subtitles OFF -> rendering clean 9:16 clip (no captions).")
            vf = _build_render_filter("", "")
        else:
            # 1. Ensure we have clip-local segments (with whatever language fields we need).
            if segments is None:
                data = transcribe_clip(None, job_dir, clip_index, log,
                                       clip_start=clip_start, clip_end=clip_end, translate=True)
                segments = data.get("segments", [])

            # 2. Resolve fonts. Hindi (Devanagari) is the one that breaks if missing.
            need_hindi = (layout == "dual") or (layout == "single" and language == "hindi")
            need_latin = (layout == "dual") or (layout == "single" and language in ("english", "hinglish")) or show_title

            hi_path, hi_family = _font_from_choice(hindi_font_choice, HINDI_FONTS, _get_hindi_font)
            la_path, la_family = _font_from_choice(english_font_choice, ENGLISH_FONTS, _get_latin_font)

            if need_hindi and not hi_path:
                log.log("     WARNING: NO Devanagari font found (bundled or system). "
                        "Hindi captions may render as empty boxes. Add a Devanagari .ttf "
                        "under ./fonts/ (e.g. NotoSansDevanagari-Bold.ttf).")
            hindi_family = hi_family or "Noto Sans Devanagari"
            latin_family = la_family or "Poppins"

            wanted_paths = []
            if need_hindi and hi_path:
                wanted_paths.append(hi_path)
            if need_latin and la_path:
                wanted_paths.append(la_path)
            fontsdir = _prepare_fontsdir(wanted_paths, job_dir)

            # 3. Build the ASS for the chosen layout/language/position/style.
            #    Write it inside the job dir (always exists, cross-platform) rather than
            #    a hardcoded "/tmp" — "/tmp" doesn't exist on Windows.
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".ass", delete=False,
                dir=job_dir, prefix=f"clip{clip_index}_", encoding="utf-8"
            ) as tmp:
                ass_path = tmp.name

            # Figure out where the actual video band sits inside the 9:16 frame, so
            # "bottom"/"top" captions can pin to the edge of the footage (just below /
            # above it) rather than the very frame edge.
            # All clips seek into the SAME source video, so its dimensions are probed
            # once by the caller and passed in — avoids an ffprobe spawn per clip.
            if src_dims and src_dims[0] and src_dims[1]:
                src_w, src_h = src_dims
            else:
                src_w, src_h = _probe_dimensions(raw_path, log)
            vbox = _video_box(src_w, src_h)
            if vbox:
                log.log(f"     Video band: y {vbox[0]:.0f}–{vbox[1]:.0f} of {SHORTS_H} "
                        f"(src {src_w}x{src_h})")

            primary, secondary, has_title = make_caption_ass(
                segments, ass_path,
                layout=layout,
                language=language,
                position=position,
                caption_style=caption_style,
                accent_color=accent_color,
                title=title,
                show_title=show_title,
                hindi_font=hindi_family,
                latin_font=latin_family,
                clip_duration=clip_duration,
                video_box=vbox,
            )
            log.log(f"     Tracks : {primary} primary cues / {secondary} secondary cues / "
                    f"title={'yes' if has_title else 'no'} (fontsdir={fontsdir or 'system'})")

            if primary == 0 and secondary == 0 and not has_title:
                log.log("     WARNING: nothing to overlay - rendering plain 9:16 video.")
                vf = _build_render_filter("", "")
            else:
                vf = _build_render_filter(ass_path, fontsdir)

        # 4. SINGLE ffmpeg pass: seek into source (-ss/-to) + scale to 9:16 + (burn).
        # Input-seek BEFORE -i is fast (keyframe seek); we re-encode anyway so accuracy
        # is preserved by -to being applied on the trimmed input.
        def _build_cmd(venc_args):
            cmd = ["ffmpeg", "-y"]
            if clip_start is not None and clip_end is not None:
                cmd += ["-ss", f"{clip_start:.3f}", "-to", f"{clip_end:.3f}"]
            cmd += ["-i", raw_path, "-vf", vf]
            cmd += venc_args
            cmd += ["-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", final_output]
            return cmd

        # Pick the fastest available encoder (NVENC -> AMF -> QSV -> libx264).
        venc_args, enc_name = providers.select_video_encoder(log)
        command = _build_cmd(venc_args)
        log.log(f"     FFmpeg ({enc_name}): {' '.join(command)}")
        result = subprocess.run(command, capture_output=True, text=True)

        # ROBUSTNESS: if a hardware encoder fails (driver/session limits/etc.), fall
        # back to the always-available CPU encoder so a clip is NEVER lost to a GPU hiccup.
        if result.returncode != 0 and enc_name != "cpu":
            log.log(f"     HW encoder '{enc_name}' failed (rc={result.returncode}); "
                    f"retrying with CPU libx264.\n     {result.stderr[-600:]}")
            command = _build_cmd(providers.CPU_ENCODER_ARGS)
            result = subprocess.run(command, capture_output=True, text=True)

        if result.returncode != 0:
            log.log(f"     FFMPEG STDERR:\n{result.stderr[-1500:]}")
            log.log(f"     FAILED (return code {result.returncode})")
            return None

        if not os.path.exists(final_output) or os.path.getsize(final_output) < 1000:
            log.log(f"     FAILED - output missing or too small")
            return None

        log.log(f"     SUCCESS -> {final_output} ({os.path.getsize(final_output)//1024} KB)")
        if clip_callback:
            clip_callback(final_output, reason)
        return final_output

    except Exception as e:
        log.error(f"Clip {clip_index} burn failed: {e}", e)
        return None

    finally:
        if ass_path:
            try: os.unlink(ass_path)
            except OSError: pass


# ─────────────────────────────────────────────────────────────
# Main entry point - subtitle burning only
# ─────────────────────────────────────────────────────────────

def execute_subtitle_workflow(
    job_dir: str,
    manifest_path: str = None,
    clip_callback=None,
    status_callback=None,
    subtitle_options: dict = None,
) -> tuple:
    """Reads clips_manifest.json and produces all finished Shorts.

    Caption behaviour comes from the manifest's "subtitle_options" block (written by
    select_clips.py), and can be overridden by passing `subtitle_options` here:
        burn_subtitles    : bool  - False => clean 9:16 clips, no captions
        subtitle_layout   : "single" | "dual"
        subtitle_language : "hindi" | "english" | "hinglish"   (single layout)
        subtitle_position : "top" | "middle" | "bottom"        (single layout)
        caption_style     : "outline" | "box"
        show_title        : bool

    STAGE 4 — transcribe each clip, then run ONLY the language pass(es) actually needed
    (translate for English/dual, transliterate for Hinglish, titles if enabled).
    STAGE 5 — render every clip in PARALLEL; each clip is a single ffmpeg pass that
    seeks into the SOURCE video (-ss/-to) and cuts + scales to 9:16 + (burns captions).

    Returns (final_clips, log_path).
    """
    import concurrent.futures, os
    if manifest_path is None:
        manifest_path = os.path.join(job_dir, "clips_manifest.json")

    log = DiagnosticLog(job_dir)
    log.section("JOB INFO")
    log.log(f"   Job dir       : {job_dir}")
    log.log(f"   Manifest path : {manifest_path}")

    final_clips = []

    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        raw_clips = manifest.get("clips", [])
        clips_dir = os.path.join(job_dir, "clips")
        os.makedirs(clips_dir, exist_ok=True)

        # ── Resolve caption config: manifest defaults, overridden by any caller args ──
        cfg = dict(manifest.get("subtitle_options", {}) or {})
        if subtitle_options:
            cfg.update({k: v for k, v in subtitle_options.items() if v is not None})

        burn          = bool(cfg.get("burn_subtitles", True))
        layout        = str(cfg.get("subtitle_layout", "single")).lower()
        language      = str(cfg.get("subtitle_language", "hindi")).lower()
        position      = str(cfg.get("subtitle_position", "bottom")).lower()
        caption_style = str(cfg.get("caption_style", "outline")).lower()
        accent_color  = str(cfg.get("caption_accent", "") or "")
        hindi_font_choice   = str(cfg.get("hindi_font", "") or "").lower()
        english_font_choice = str(cfg.get("english_font", "") or "").lower()
        show_title    = bool(cfg.get("show_title", False))
        if layout not in ("single", "dual"):
            layout = "single"
        if language not in ("hindi", "english", "hinglish"):
            language = "hindi"
        if position not in ("top", "middle", "bottom", "below"):
            position = "bottom"
        if caption_style not in VALID_CAPTION_STYLES:
            caption_style = "outline"
        if hindi_font_choice and hindi_font_choice not in VALID_HINDI_FONTS:
            hindi_font_choice = ""
        if english_font_choice and english_font_choice not in VALID_ENGLISH_FONTS:
            english_font_choice = ""

        log.section("CAPTION CONFIG")
        log.log(f"   burn={burn} | layout={layout} | language={language} | position={position} | "
                f"style={caption_style} | accent={accent_color or 'default'} | title={show_title}")
        log.log(f"   fonts: hindi={hindi_font_choice or 'auto'} | english={english_font_choice or 'auto'}")

        clip_segments = {}   # index -> segments list
        clip_titles = {}     # index -> title string

        if not burn:
            # No captions at all — skip transcription/translation entirely (saves API $).
            log.section("STAGE 4 - SKIPPED (subtitles off)")
            log.log("   Subtitles are OFF — clips will be cut + scaled to 9:16 with no captions.")
            if status_callback:
                status_callback("Rendering clean 9:16 clips (no subtitles)...")
        else:
            # Subtitle source engine:
            #   "deepgram" (DEFAULT) — transcribe each SELECTED clip with Deepgram nova-3.
            #                          Best Hindi/Hinglish word-level accuracy. Used when a
            #                          DEEPGRAM_API_KEY is set; per-clip failures fall back
            #                          to whisper-slice automatically so captions never blank.
            #   "whisper"            — reuse the full-video Whisper transcript by slicing it
            #                          to each clip (free, no extra API, needs GROQ).
            engine = os.environ.get("SUBTITLE_ENGINE", "deepgram").lower()
            whisper_full = os.path.join(job_dir, "transcript_full.json")
            have_dg_key = bool((os.environ.get("DEEPGRAM_API_KEY") or "").strip())
            have_whisper = os.path.exists(whisper_full)
            # Deepgram only when explicitly requested AND a real key exists.
            if engine == "deepgram" and not have_dg_key:
                log.log("   SUBTITLE_ENGINE=deepgram but no DEEPGRAM_API_KEY -> using whisper-slice")
                engine = "whisper"
            # Whisper needs the full transcript; if it's somehow missing but a Deepgram
            # key is available, use Deepgram as the fallback instead.
            if engine == "whisper" and not have_whisper:
                if have_dg_key:
                    log.log("   No full transcript found -> falling back to Deepgram")
                    engine = "deepgram"
                else:
                    log.log("   WARNING: no full transcript and no Deepgram key -> captions may be empty")
            log.log(f"   Subtitle engine: {engine}")

            # ── STAGE 4: per-clip transcription -> ONLY the needed language pass(es) ──
            log.section("STAGE 4 - PER-CLIP TRANSCRIBE + LANGUAGE PREP")
            if status_callback:
                status_callback("Preparing subtitles for all clips...")

            all_segment_lists = []
            order = []
            for rc in raw_clips:
                idx = rc["index"]
                cs, ce = rc.get("start"), rc.get("end")
                segs = []
                try:
                    if engine == "deepgram":
                        # Cut just this clip's AUDIO from the source, transcribe it.
                        try:
                            clip_audio = os.path.join(clips_dir, f"clip_{idx}_audio.mp3")
                            _extract_clip_audio(rc["raw_path"], clip_audio, cs, ce, log)
                            segs = _deepgram_transcribe(clip_audio, log).get("segments", [])
                            try: os.remove(clip_audio)
                            except OSError: pass
                        except Exception as dg_err:
                            # ROBUSTNESS: a Deepgram 500/timeout on one clip must not blank
                            # its captions — fall back to slicing the full Whisper transcript.
                            log.log(f"   Clip {idx}: Deepgram failed ({dg_err}); "
                                    f"falling back to whisper-slice")
                            segs = []
                        if not segs and os.path.exists(whisper_full) and cs is not None and ce is not None:
                            segs = _slice_transcript(whisper_full, cs, ce).get("segments", [])
                    elif os.path.exists(whisper_full) and cs is not None and ce is not None:
                        segs = _slice_transcript(whisper_full, cs, ce).get("segments", [])
                except Exception as e:
                    log.error(f"   Clip {idx}: transcription failed: {e}", e)
                    segs = []
                clip_segments[idx] = segs
                all_segment_lists.append(segs)
                order.append(idx)

            # Run ONLY the passes the chosen captions require (each is one Groq call):
            need_translate = (layout == "dual") or (layout == "single" and language == "english")
            need_translit  = (layout == "single" and language == "hinglish")

            if need_translate:
                batch_translate_clips(all_segment_lists, log)
            if need_translit:
                batch_transliterate_clips(all_segment_lists, log)
            if show_title:
                titles = batch_generate_titles(all_segment_lists, log)
                clip_titles = {order[i]: titles[i] for i in range(len(order))}

            # Persist a single combined transcript file for reference.
            try:
                transcripts_txt = os.path.join(job_dir, "clip_transcripts.txt")
                with open(transcripts_txt, "w", encoding="utf-8") as tf:
                    for idx in order:
                        segs = clip_segments[idx]
                        tf.write("=" * 70 + f"\nCLIP {idx}  ({len(segs)} segments)\n")
                        if show_title:
                            tf.write(f"TITLE: {clip_titles.get(idx,'')}\n")
                        tf.write("=" * 70 + "\n")
                        for seg in segs:
                            tf.write(f"[{seg.get('start',0):7.2f} - {seg.get('end',0):7.2f}]\n")
                            tf.write(f"   HI: {seg.get('text','').strip()}\n")
                            if seg.get("text_en"):
                                tf.write(f"   EN: {seg.get('text_en','').strip()}\n")
                            if seg.get("text_hinglish"):
                                tf.write(f"   HG: {seg.get('text_hinglish','').strip()}\n")
                        tf.write("\n")
            except Exception as e:
                log.error(f"Could not write combined transcript file: {e}", e)

        # ── STAGE 5: parallel burn (single pass per clip, seek into source) ──
        log.section("STAGE 5 - PARALLEL RENDER (cut + scale + optional captions)")
        log.log(f"   Total clips: {len(raw_clips)}")

        # Probe the SOURCE video's dimensions ONCE — every clip seeks into the same
        # file, so a single ffprobe replaces one-per-clip.
        src_video = manifest.get("video_path") or (raw_clips[0]["raw_path"] if raw_clips else None)
        src_dims = _probe_dimensions(src_video, log) if src_video else (None, None)
        log.log(f"   Source dimensions: {src_dims[0]}x{src_dims[1]}" if src_dims[0] else
                "   Source dimensions: unknown (will probe per clip)")

        # Decide which encoder we'll use so we can size the worker pool sensibly.
        _venc_args, _enc_name = providers.select_video_encoder(log)
        _env_workers = os.environ.get("MAX_RENDER_WORKERS")
        if _env_workers:
            max_workers = max(1, int(_env_workers))
        elif _enc_name != "cpu":
            # Hardware encoders share ONE GPU encode block; a few parallel ffmpegs keep
            # it fed (CPU still does libass + scaling) without thrashing the GPU session.
            max_workers = max(1, min(4, len(raw_clips)))
        else:
            try:
                import psutil
                free_gb = psutil.virtual_memory().available / (1024 ** 3)
                cores = psutil.cpu_count(logical=False) or (os.cpu_count() or 2)
                # Each CPU ffmpeg uses ~2 threads, so don't exceed physical cores.
                max_workers = max(1, min(cores, int(free_gb // 1.5)))
            except Exception:
                max_workers = 2   # safe default for low-RAM machines (e.g. 4GB WSL)
        log.log(f"   Encoder: {_enc_name} | Parallel workers: {max_workers}")

        results = {}
        done_count = {"n": 0}

        def _process(rc):
            idx = rc["index"]
            out = burn_subtitles_for_clip(
                rc["raw_path"], idx, job_dir, clips_dir, log,
                clip_callback=clip_callback, reason=rc.get("reason", ""),
                clip_start=rc.get("start"), clip_end=rc.get("end"),
                segments=clip_segments.get(idx),
                title=clip_titles.get(idx, ""),
                burn=burn,
                layout=layout,
                language=language,
                position=position,
                caption_style=caption_style,
                accent_color=accent_color,
                hindi_font_choice=hindi_font_choice,
                english_font_choice=english_font_choice,
                show_title=show_title,
                src_dims=src_dims,
            )
            done_count["n"] += 1
            if status_callback:
                status_callback(f"Rendered {done_count['n']}/{len(raw_clips)} clips...")
            return idx, out

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(_process, rc): rc["index"] for rc in raw_clips}
            for fut in concurrent.futures.as_completed(futures):
                try:
                    idx, out = fut.result()
                    results[idx] = out
                except Exception as e:
                    log.error(f"Worker raised exception: {e}", e)

        for rc in raw_clips:
            out = results.get(rc["index"])
            if out:
                final_clips.append(out)

    except Exception as e:
        log.section("SUBTITLE PIPELINE CRASHED")
        log.error(f"Unhandled exception: {e}", e)
        final_clips = []

    log.finalize(final_clips)
    return final_clips, log.path


# ─────────────────────────────────────────────────────────────
# CLI usage
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Burn word-by-word subtitles onto raw clips produced by select_clips.py"
    )
    parser.add_argument(
        "job_dir",
        help="Path to the job directory produced by select_clips.py "
             "(e.g. output/<job_id>), containing clips_manifest.json"
    )
    parser.add_argument(
        "--manifest",
        default=None,
        help="Optional explicit path to clips_manifest.json (defaults to <job_dir>/clips_manifest.json)"
    )
    parser.add_argument("--no-burn", action="store_true",
                        help="Render clean 9:16 clips with NO captions (overrides the manifest)")
    parser.add_argument("--layout", choices=["single", "dual"], default=None,
                        help="single = one language track; dual = Hindi top + English bottom")
    parser.add_argument("--language", choices=["hindi", "english", "hinglish"], default=None,
                        help="Caption language for single layout")
    parser.add_argument("--position", choices=["top", "middle", "bottom", "below"], default=None,
                        help="bottom = on the video; below = under the video (in the letterbox bar)")
    parser.add_argument("--style",
                        choices=list(VALID_CAPTION_STYLES),
                        default=None,
                        help="Caption look: " + ", ".join(VALID_CAPTION_STYLES))
    parser.add_argument("--accent", default=None,
                        help="Accent colour for karaoke/word_pop active word, e.g. '#FFE600'")
    parser.add_argument("--title", action="store_true", default=None,
                        help="Overlay an AI-generated headline title")
    args = parser.parse_args()

    overrides = {}
    if args.no_burn:        overrides["burn_subtitles"] = False
    if args.layout:         overrides["subtitle_layout"] = args.layout
    if args.language:       overrides["subtitle_language"] = args.language
    if args.position:       overrides["subtitle_position"] = args.position
    if args.style:          overrides["caption_style"] = args.style
    if args.accent:         overrides["caption_accent"] = args.accent
    if args.title:          overrides["show_title"] = True

    clips, log_path = execute_subtitle_workflow(
        args.job_dir, manifest_path=args.manifest,
        subtitle_options=overrides or None,
    )

    print(f"\nDone. {len(clips)} subtitled clip(s) produced.")
    print(f"Log: {log_path}")