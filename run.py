#!/usr/bin/env python3
"""
run.py — one command to set up AND start ShortsAI.

It creates a virtual environment, installs everything in requirements.txt, checks for
ffmpeg, wires up templates/index.html, makes sure a .env exists, and launches the web
app — so you never have to type the long uvicorn command or set things up by hand.

    python run.py                 # first run: set everything up, then start the server
    python run.py --reload        # dev mode: auto-restart when you edit code
    python run.py --port 9000     # use a different port
    python run.py --host 0.0.0.0  # expose on your LAN (open it from your phone)
    python run.py --os wsl        # force the OS (auto-detected by default)
    python run.py --reinstall     # re-run pip install (after editing requirements.txt)
    python run.py --install-ffmpeg# also try to install ffmpeg (Linux/WSL via apt, mac via brew)
    python run.py --setup-only    # set up but don't start the server

Works on Windows, WSL, Linux and macOS. The only prerequisite is Python 3.10+.
(On Windows run it with `python run.py`; on Linux/WSL/mac use `python3 run.py`.)
"""

import os
import sys
import shutil
import argparse
import platform
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
VENV = HERE / ".venv"
STAMP = VENV / ".installed"           # marker so we don't reinstall every launch
REQS = HERE / "requirements.txt"


# ── helpers ───────────────────────────────────────────────────────────────
def c(text, code):
    """Colourise (skipped automatically if the terminal can't handle it)."""
    if os.environ.get("NO_COLOR") or not sys.stdout.isatty():
        return text
    return f"\033[{code}m{text}\033[0m"


def info(msg):  print(c("• ", "36") + msg)
def ok(msg):    print(c("✓ ", "32") + msg)
def warn(msg):  print(c("! ", "33") + msg)
def err(msg):   print(c("✗ ", "31") + msg)


def venv_python() -> Path:
    return VENV / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def detect_os() -> str:
    if os.name == "nt":
        return "windows"
    system = platform.system().lower()
    if system == "darwin":
        return "mac"
    try:
        if "microsoft" in Path("/proc/version").read_text().lower():
            return "wsl"
    except Exception:
        pass
    return "linux"


def have(cmd) -> bool:
    return shutil.which(cmd) is not None


def run(cmd, check=True, **kw):
    print("  " + c("$ " + " ".join(str(x) for x in cmd), "90"))
    return subprocess.run(cmd, check=check, **kw)


# ── steps ─────────────────────────────────────────────────────────────────
def check_python():
    if sys.version_info < (3, 10):
        warn(f"Python {sys.version_info.major}.{sys.version_info.minor} detected; "
             "3.10+ is recommended. Continuing anyway.")
    else:
        ok(f"Python {sys.version_info.major}.{sys.version_info.minor}")


def ensure_venv():
    if venv_python().exists():
        ok("Virtual environment ready (.venv)")
        return False           # already existed
    info("Creating virtual environment in .venv ...")
    import venv as _venv
    _venv.EnvBuilder(with_pip=True).create(VENV)
    ok("Virtual environment created")
    return True                # freshly created


def install_deps(force=False):
    if not REQS.exists():
        err(f"requirements.txt not found next to run.py ({REQS}).")
        sys.exit(1)
    if STAMP.exists() and not force:
        ok("Dependencies already installed (use --reinstall to refresh)")
        return
    vpy = venv_python()
    info("Installing dependencies (this can take a minute the first time) ...")
    try:
        run([str(vpy), "-m", "pip", "install", "--upgrade", "pip"])
        run([str(vpy), "-m", "pip", "install", "-r", str(REQS)])
    except subprocess.CalledProcessError:
        err("pip install failed. Check your internet connection and try again "
            "(or re-run with --reinstall).")
        sys.exit(1)
    STAMP.write_text("ok\n")
    ok("Dependencies installed")


def ensure_ffmpeg(target_os, try_install):
    if have("ffmpeg") and have("ffprobe"):
        ok("ffmpeg found")
        return
    warn("ffmpeg / ffprobe not found — it is REQUIRED to cut and caption videos.")
    if try_install:
        try:
            if target_os in ("linux", "wsl"):
                info("Attempting: sudo apt-get install -y ffmpeg")
                run(["sudo", "apt-get", "update"])
                run(["sudo", "apt-get", "install", "-y", "ffmpeg"])
            elif target_os == "mac" and have("brew"):
                run(["brew", "install", "ffmpeg"])
            elif target_os == "windows" and have("winget"):
                run(["winget", "install", "--id", "Gyan.FFmpeg", "-e"])
            if have("ffmpeg"):
                ok("ffmpeg installed")
                return
        except Exception:
            pass
    # Print manual instructions per OS.
    hints = {
        "linux": "sudo apt-get install -y ffmpeg",
        "wsl":   "sudo apt-get install -y ffmpeg",
        "mac":   "brew install ffmpeg",
        "windows": "winget install Gyan.FFmpeg     (or: choco install ffmpeg)",
    }
    warn("Install ffmpeg, then re-run.  ->  " + hints.get(target_os, "install ffmpeg"))
    warn("(The server will still start so you can see the UI, but rendering needs ffmpeg.)")


def ensure_templates():
    tdir = HERE / "templates"
    tindex = tdir / "index.html"
    src = HERE / "index.html"
    if tindex.exists():
        ok("templates/index.html in place")
        return
    if src.exists():
        tdir.mkdir(exist_ok=True)
        shutil.copy2(src, tindex)
        ok("Copied index.html -> templates/index.html")
    else:
        warn("index.html not found; the web UI may not load until it's at templates/index.html")


def ensure_env():
    env = HERE / ".env"
    if not env.exists():
        env.write_text(
            "# Fill these in. GROQ is required; DEEPGRAM is optional.\n"
            "GROQ_API_KEY=\n"
            "DEEPGRAM_API_KEY=\n"
        )
        warn("Created a blank .env — add your GROQ_API_KEY before processing videos.")
        return
    text = env.read_text()
    has_key = any(line.startswith("GROQ_API_KEY=") and line.strip() != "GROQ_API_KEY="
                  for line in text.splitlines())
    if has_key:
        ok(".env found (GROQ_API_KEY set)")
    else:
        warn(".env has no GROQ_API_KEY value yet — add it before processing videos.")


def ensure_font():
    # Devanagari is only needed for Hindi / dual captions; warn softly.
    bundled = list((HERE / "fonts").glob("**/*Devanagari*.ttf")) if (HERE / "fonts").exists() else []
    if bundled:
        ok("Devanagari font found in ./fonts")
        return
    common = [
        "/usr/share/fonts/truetype/noto/NotoSansDevanagari-Bold.ttf",
        "/usr/share/fonts/truetype/lohit-devanagari/Lohit-Devanagari.ttf",
    ]
    if any(Path(p).exists() for p in common):
        ok("Devanagari system font found")
    else:
        warn("No Devanagari font found — Hindi/dual captions need one. "
             "Drop NotoSansDevanagari-Bold.ttf into ./fonts/ "
             "(English/Hinglish/no-caption modes work without it).")


def launch(host, port, reload_):
    vpy = venv_python()
    cmd = [str(vpy), "-m", "uvicorn", "app:app", "--host", host, "--port", str(port)]
    if reload_:
        cmd.append("--reload")
    shown = host if host != "0.0.0.0" else "localhost"
    print()
    ok(f"Starting ShortsAI  ->  http://{shown}:{port}")
    info("Press Ctrl+C to stop.\n")
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print()
        info("Stopped.")


# ── main ──────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Set up and run ShortsAI with one command.")
    ap.add_argument("--os", choices=["auto", "windows", "wsl", "linux", "mac"], default="auto",
                    help="Override OS detection (used for ffmpeg install hints).")
    ap.add_argument("--host", default="127.0.0.1", help="Bind host (use 0.0.0.0 for LAN).")
    ap.add_argument("--port", type=int, default=8000, help="Port (default 8000).")
    ap.add_argument("--reload", action="store_true", help="Dev mode: auto-restart on edits.")
    ap.add_argument("--reinstall", action="store_true", help="Force a fresh pip install.")
    ap.add_argument("--skip-install", action="store_true", help="Skip dependency install.")
    ap.add_argument("--install-ffmpeg", action="store_true", help="Try to install ffmpeg too.")
    ap.add_argument("--setup-only", action="store_true", help="Set up but don't start the server.")
    args = ap.parse_args()

    os.chdir(HERE)
    target_os = detect_os() if args.os == "auto" else args.os

    print(c("\n  ShortsAI launcher", "1") + c(f"   ({target_os})\n", "90"))

    check_python()
    fresh = ensure_venv()
    if not args.skip_install:
        install_deps(force=args.reinstall or fresh and not STAMP.exists())
    else:
        warn("Skipping dependency install (--skip-install)")
    ensure_ffmpeg(target_os, args.install_ffmpeg)
    ensure_templates()
    ensure_env()
    ensure_font()

    if args.setup_only:
        print()
        ok("Setup complete. Start it any time with:  " + c("python run.py", "1"))
        return

    launch(args.host, args.port, args.reload)


if __name__ == "__main__":
    main()
