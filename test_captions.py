#!/usr/bin/env python3
"""
test_captions.py — preview caption STYLES / POSITIONS without the server or any API keys.

Why: when you're iterating on how captions look, you don't want to run the whole
pipeline (download → Whisper → LLM select → Deepgram) just to see a font change. This
script calls the renderer directly on ONE local clip and produces:
    caption_preview/clip_<style>_<position>.mp4   (the rendered Short)
    caption_preview/frame_<style>_<position>.png  (a still you can glance at)

So your loop becomes:  drop in the new burn_subtitles.py  →  run this  →  look.

Usage
-----
    python test_captions.py myclip.mp4
    python test_captions.py myclip.mp4 --styles karaoke,word_pop --positions bottom,top
    python test_captions.py myclip.mp4 --language english --text "This is my caption line"
    python test_captions.py myclip.mp4 --layout dual
    python test_captions.py myclip.mp4 --styles karaoke --accent "#2EE640"

Notes
-----
  * Needs ffmpeg + ffprobe on PATH (same as the pipeline).
  * Does NOT need GROQ_API_KEY / DEEPGRAM_API_KEY — those SDKs are stubbed and the
    script feeds in synthetic transcript segments. You are testing the RENDERING
    (styles / positions / layout / placement), which is what changes most.
  * Default language is English so it renders even without a Devanagari font installed.
    Use --language hindi to test the Devanagari word-by-word path (needs a Hindi font).
"""

import os
import sys
import types
import argparse
import subprocess


# ── Make burn_subtitles importable even if the API SDKs aren't installed ──
def _ensure_stub(mod_name, attrs):
    try:
        __import__(mod_name)
    except Exception:
        m = types.ModuleType(mod_name)
        for name in attrs:
            setattr(m, name, type(name, (), {"__init__": lambda self, *a, **k: None}))
        sys.modules[mod_name] = m


_ensure_stub("groq", ["Groq"])
_ensure_stub("deepgram", ["DeepgramClient"])

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

try:
    import burn_subtitles as B
except Exception as e:
    print(f"ERROR: could not import burn_subtitles.py from {HERE}\n  {e}")
    sys.exit(1)


def _probe_duration(path):
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", path],
            capture_output=True, text=True)
        return float(r.stdout.strip())
    except Exception:
        return None


def _make_segments(text, duration):
    """Build a few synthetic transcript segments spanning the clip, with all language
    fields + fabricated Devanagari word timings (so the karaoke real-timing path works)."""
    words = text.split()
    if not words:
        words = ["Your", "caption", "goes", "here"]
    # split into 2 roughly-equal segments across the clip
    mid = max(1, len(words) // 2)
    groups = [words[:mid], words[mid:]] if len(words) > 3 else [words]
    segs = []
    span = duration / len(groups)
    for gi, g in enumerate(groups):
        s = gi * span
        e = (gi + 1) * span
        line = " ".join(g)
        wlist = []
        if g:
            wspan = (e - s) / len(g)
            for wi, w in enumerate(g):
                wlist.append({"word": w, "start": s + wi * wspan, "end": s + (wi + 1) * wspan})
        segs.append({"start": s, "end": e, "text": line,
                     "text_en": line, "text_hinglish": line, "words": wlist})
    return segs


def main():
    ap = argparse.ArgumentParser(description="Preview caption styles/positions on a local clip.")
    ap.add_argument("video", help="path to a local video clip to caption")
    ap.add_argument("--styles", default="outline,box,white_box,bold_yellow,karaoke,word_pop",
                    help="comma list: outline,box,white_box,bold_yellow,karaoke,word_pop")
    ap.add_argument("--positions", default="bottom",
                    help="comma list: top,middle,bottom (on video),below (under video)")
    ap.add_argument("--language", default="english", choices=["hindi", "english", "hinglish"])
    ap.add_argument("--layout", default="single", choices=["single", "dual"])
    ap.add_argument("--accent", default="#FFE600", help="accent colour for karaoke/word_pop")
    ap.add_argument("--text", default="This changes everything you need to watch till the end",
                    help="caption text to display")
    ap.add_argument("--title", action="store_true", help="also overlay the AI title line")
    ap.add_argument("--out", default="caption_preview", help="output folder")
    args = ap.parse_args()

    if not os.path.exists(args.video):
        print(f"ERROR: file not found: {args.video}")
        sys.exit(1)

    duration = _probe_duration(args.video) or 5.0
    clip_len = min(duration, 6.0)   # preview only the first few seconds
    segs = _make_segments(args.text, clip_len)

    out_dir = os.path.abspath(args.out)
    clips_dir = os.path.join(out_dir, "clips")
    os.makedirs(clips_dir, exist_ok=True)
    log = B.DiagnosticLog(out_dir)

    styles = [s.strip() for s in args.styles.split(",") if s.strip()]
    positions = [p.strip() for p in args.positions.split(",") if p.strip()]

    print(f"\nPreviewing '{os.path.basename(args.video)}'  ({clip_len:.1f}s)  "
          f"lang={args.language} layout={args.layout}")
    print(f"styles={styles}  positions={positions}\n" + "-" * 60)

    idx = 0
    made = []
    for style in styles:
        for pos in positions:
            idx += 1
            out = B.burn_subtitles_for_clip(
                args.video, idx, out_dir, clips_dir, log,
                clip_start=0.0, clip_end=clip_len, segments=segs,
                layout=args.layout, language=args.language,
                position=pos, caption_style=style, accent_color=args.accent,
                show_title=args.title,
            )
            if not out or not os.path.exists(out):
                print(f"  FAIL  {style:12s} {pos:6s}")
                continue
            tag = f"{style}_{pos}"
            clip_dst = os.path.join(out_dir, f"clip_{tag}.mp4")
            os.replace(out, clip_dst)
            frame_dst = os.path.join(out_dir, f"frame_{tag}.png")
            subprocess.run(["ffmpeg", "-y", "-i", clip_dst, "-ss", f"{clip_len * 0.45:.2f}",
                            "-vframes", "1", frame_dst], capture_output=True)
            made.append((tag, clip_dst, frame_dst))
            print(f"  OK    {style:12s} {pos:6s} -> clip_{tag}.mp4 + frame_{tag}.png")

    # Optional: stitch the preview frames into one labelled contact sheet.
    if made:
        try:
            seqdir = os.path.join(out_dir, "_thumbs")
            os.makedirs(seqdir, exist_ok=True)
            la_path, _ = B._get_latin_font()
            for i, (tag, _clip, frame) in enumerate(made, start=1):
                if not os.path.exists(frame):
                    continue
                thumb = os.path.join(seqdir, f"img_{i:04d}.png")
                vf = "scale=360:-1"
                if la_path:
                    fp = la_path.replace("\\", "/").replace(":", "\\:")
                    vf += (f",drawtext=fontfile='{fp}':text='{tag}':x=12:y=12:fontsize=22:"
                           f"fontcolor=white:box=1:boxcolor=0x000000AA:boxborderw=8")
                subprocess.run(["ffmpeg", "-y", "-i", frame, "-vf", vf, thumb],
                               capture_output=True)
            n = sum(1 for _ in os.listdir(seqdir))
            if n:
                cols = min(3, n)
                rows = (n + cols - 1) // cols
                sheet = os.path.join(out_dir, "_contact_sheet.png")
                subprocess.run(
                    ["ffmpeg", "-y", "-framerate", "1",
                     "-i", os.path.join(seqdir, "img_%04d.png"),
                     "-frames:v", "1",
                     "-vf", f"tile={cols}x{rows}:margin=10:padding=10:color=0x1a1a1a",
                     sheet],
                    capture_output=True)
                if os.path.exists(sheet):
                    print(f"\nContact sheet (all variants in one image): {sheet}")
        except Exception:
            pass

    print("-" * 60)
    print(f"Done. {len(made)} preview(s) in: {out_dir}")
    print("Open the frame_*.png files (or _contact_sheet.png) to eyeball the looks.")


if __name__ == "__main__":
    main()
