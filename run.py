#!/usr/bin/env python3
"""
run.py — one command to set up AND start ShortsAI.

Detects the OS/distro and installs everything automatically:
  • Python deps via pip  (fastapi, uvicorn, groq, yt-dlp, …)
  • ffmpeg with libx264  via the system package manager
  • All 18 caption fonts  downloaded into ./fonts/

Usage:
    python3 run.py                   # first-time setup + start the server
    python3 run.py --reload          # dev mode: auto-restart on code changes
    python3 run.py --port 9000       # use a different port
    python3 run.py --host 0.0.0.0    # expose on your LAN / phone
    python3 run.py --reinstall       # force re-run pip install
    python3 run.py --install-ffmpeg  # auto-install/fix ffmpeg
    python3 run.py --skip-fonts      # skip font downloads (faster re-runs)
    python3 run.py --no-sudo         # skip every step that needs sudo
    python3 run.py --setup-only      # set up without starting the server
    python3 run.py --skip-test       # skip API key + subtitle engine tests

Supported platforms:
  Fedora  — dnf + RPM Fusion (adds libx264)
  Ubuntu / Debian / WSL — apt-get
  Arch / Manjaro — pacman
  openSUSE — zypper
  macOS — Homebrew
  Windows — winget / chocolatey / manual
"""

import os
import sys
import shutil
import argparse
import platform
import subprocess
import urllib.request
import urllib.error
from pathlib import Path

HERE      = Path(__file__).resolve().parent
VENV      = HERE / ".venv"
STAMP     = VENV / ".installed"
REQS      = HERE / "requirements.txt"
FONTS_DIR = HERE / "fonts"

# ── Font catalogue ─────────────────────────────────────────────────────────────
FONT_URLS = {
    # Hindi / Devanagari
    "NotoSansDevanagari-Bold.ttf":
        "https://github.com/notofonts/devanagari/raw/main/fonts/NotoSansDevanagari/hinted/ttf/NotoSansDevanagari-Bold.ttf",
    "Mukta-Bold.ttf":
        "https://github.com/google/fonts/raw/main/ofl/mukta/Mukta-Bold.ttf",
    "Hind-Bold.ttf":
        "https://github.com/google/fonts/raw/main/ofl/hind/Hind-Bold.ttf",
    "RozhaOne-Regular.ttf":
        "https://github.com/google/fonts/raw/main/ofl/rozhaone/RozhaOne-Regular.ttf",
    "Kalam-Bold.ttf":
        "https://github.com/google/fonts/raw/main/ofl/kalam/Kalam-Bold.ttf",
    "Baloo2-Bold.ttf":
        "https://github.com/google/fonts/raw/main/ofl/baloo2/Baloo2-Bold.ttf",
    "Laila-Bold.ttf":
        "https://github.com/google/fonts/raw/main/ofl/laila/Laila-Bold.ttf",
    "Rajdhani-Bold.ttf":
        "https://github.com/google/fonts/raw/main/ofl/rajdhani/Rajdhani-Bold.ttf",
    # English / Latin
    "Poppins-Bold.ttf":
        "https://github.com/google/fonts/raw/main/ofl/poppins/Poppins-Bold.ttf",
    "Anton-Regular.ttf":
        "https://github.com/google/fonts/raw/main/ofl/anton/Anton-Regular.ttf",
    "BebasNeue-Regular.ttf":
        "https://github.com/google/fonts/raw/main/ofl/bebasneue/BebasNeue-Regular.ttf",
    "ArchivoBlack-Regular.ttf":
        "https://github.com/google/fonts/raw/main/ofl/archivoblack/ArchivoBlack-Regular.ttf",
    "FjallaOne-Regular.ttf":
        "https://github.com/google/fonts/raw/main/ofl/fjallaone/FjallaOne-Regular.ttf",
    "Oswald-Bold.ttf":
        "https://github.com/google/fonts/raw/main/ofl/oswald/Oswald-Bold.ttf",
    "Montserrat-Bold.ttf":
        "https://github.com/google/fonts/raw/main/ofl/montserrat/Montserrat-Bold.ttf",
    "Staatliches-Regular.ttf":
        "https://github.com/google/fonts/raw/main/ofl/staatliches/Staatliches-Regular.ttf",
    "BarlowCondensed-Bold.ttf":
        "https://github.com/google/fonts/raw/main/ofl/barlowcondensed/BarlowCondensed-Bold.ttf",
    "Righteous-Regular.ttf":
        "https://github.com/google/fonts/raw/main/ofl/righteous/Righteous-Regular.ttf",
}

# ── Colours ────────────────────────────────────────────────────────────────────
def c(text, code):
    if os.environ.get("NO_COLOR") or not sys.stdout.isatty():
        return text
    return f"\033[{code}m{text}\033[0m"

def info(msg):  print(c("• ", "36") + msg)
def ok(msg):    print(c("✓ ", "32") + msg)
def warn(msg):  print(c("! ", "33") + msg)
def err(msg):   print(c("✗ ", "31") + msg, file=sys.stderr)
def hdr(msg):   print(c(f"\n── {msg} ", "1") + c("─" * max(0, 50 - len(msg)), "90"))

# ── Shell helpers ──────────────────────────────────────────────────────────────
def run_cmd(cmd, check=True, **kw):
    print("  " + c("$ " + " ".join(str(x) for x in cmd), "90"))
    return subprocess.run(cmd, check=check, **kw)

def have(cmd):
    return shutil.which(cmd) is not None

def venv_python():
    return VENV / ("Scripts/python.exe" if os.name == "nt" else "bin/python")

def venv_bin(name):
    suffix = ".exe" if os.name == "nt" else ""
    scripts = "Scripts" if os.name == "nt" else "bin"
    return VENV / scripts / (name + suffix)

# ── OS / distro detection ──────────────────────────────────────────────────────
def _read_os_release():
    for path in ("/etc/os-release", "/usr/lib/os-release"):
        try:
            return Path(path).read_text().lower()
        except Exception:
            pass
    return ""

def _detect_distro():
    text = _read_os_release()
    if "fedora" in text:
        return "fedora"
    if "arch" in text or "manjaro" in text or "endeavour" in text or "garuda" in text:
        return "arch"
    if "ubuntu" in text or "debian" in text or "mint" in text or "pop!" in text:
        return "debian"
    if "opensuse" in text or "suse" in text:
        return "opensuse"
    if "rhel" in text or "centos" in text or "rocky" in text or "almalinux" in text:
        return "rhel"
    return "linux"

def detect_os():
    """Return (os_type, distro). distro is only meaningful on Linux/WSL."""
    if os.name == "nt":
        return "windows", ""
    if platform.system().lower() == "darwin":
        return "mac", ""
    try:
        if "microsoft" in Path("/proc/version").read_text().lower():
            return "wsl", _detect_distro()
    except Exception:
        pass
    return "linux", _detect_distro()

# ── Python version ─────────────────────────────────────────────────────────────
def check_python():
    vi = sys.version_info
    if vi < (3, 10):
        warn(f"Python {vi.major}.{vi.minor} detected — 3.10+ is recommended")
    else:
        ok(f"Python {vi.major}.{vi.minor}.{vi.micro}")

# ── Virtual environment ────────────────────────────────────────────────────────
def ensure_venv():
    if venv_python().exists():
        ok("Virtual environment ready  (.venv)")
        return False
    info("Creating virtual environment in .venv …")
    import venv as _venv
    _venv.EnvBuilder(with_pip=True).create(VENV)
    ok("Virtual environment created")
    return True

# ── Python dependencies ────────────────────────────────────────────────────────
def _reqs_fingerprint() -> str:
    """Hash of requirements.txt, recorded in the install stamp."""
    import hashlib
    return hashlib.sha256(REQS.read_bytes()).hexdigest()[:16]


def install_deps(force=False):
    if not REQS.exists():
        err(f"requirements.txt not found at {REQS}")
        sys.exit(1)
    # The stamp stores WHICH requirements.txt was installed, not just "done". So
    # when a dependency is added (e.g. google-genai for the Gemini best-clips mode),
    # an existing .venv re-runs pip instead of silently skipping the new package.
    fingerprint = _reqs_fingerprint()
    if STAMP.exists() and not force and STAMP.read_text().strip() == fingerprint:
        ok("Python dependencies already installed  (--reinstall to refresh)")
        return
    vpy = str(venv_python())
    if STAMP.exists():
        info("requirements.txt changed since last install — updating dependencies …")
    else:
        info("Installing Python dependencies (first run may take a minute) …")
    try:
        run_cmd([vpy, "-m", "pip", "install", "--upgrade", "pip", "--quiet"])
        run_cmd([vpy, "-m", "pip", "install", "-r", str(REQS)])
    except subprocess.CalledProcessError:
        err("pip install failed — check your internet connection and try again.")
        sys.exit(1)
    STAMP.write_text(fingerprint + "\n")
    ok("Python dependencies installed")

def check_ytdlp():
    if venv_bin("yt-dlp").exists():
        ok("yt-dlp ready  (venv)")
    elif have("yt-dlp"):
        ok("yt-dlp ready  (system)")
    else:
        warn("yt-dlp not found in venv — YouTube downloads may fail (run after pip install)")

# ── ffmpeg ─────────────────────────────────────────────────────────────────────
def _has_libx264():
    try:
        result = subprocess.run(
            ["ffmpeg", "-encoders"],
            capture_output=True, text=True
        )
        return "libx264" in (result.stdout + result.stderr)
    except Exception:
        return False

def _ffmpeg_ok():
    return have("ffmpeg") and have("ffprobe") and _has_libx264()

def _rpm_fusion_installed():
    try:
        return subprocess.run(
            ["rpm", "-q", "rpmfusion-free-release"],
            capture_output=True
        ).returncode == 0
    except Exception:
        return False

def _ffmpeg_free_installed():
    try:
        return subprocess.run(
            ["rpm", "-q", "ffmpeg-free"],
            capture_output=True
        ).returncode == 0
    except Exception:
        return False

def _install_ffmpeg_fedora(sudo):
    if not sudo:
        warn("Need sudo to install ffmpeg on Fedora. Install manually:")
        warn("  sudo dnf install -y https://mirrors.rpmfusion.org/free/fedora/"
             "rpmfusion-free-release-$(rpm -E %fedora).noarch.rpm")
        warn("  sudo dnf swap -y ffmpeg-free ffmpeg --allowerasing")
        return False
    try:
        if not _rpm_fusion_installed():
            info("  Enabling RPM Fusion free repository …")
            fedora_ver = subprocess.check_output(
                ["rpm", "-E", "%fedora"], text=True
            ).strip()
            url = (f"https://mirrors.rpmfusion.org/free/fedora/"
                   f"rpmfusion-free-release-{fedora_ver}.noarch.rpm")
            run_cmd(["sudo", "dnf", "install", "-y", url])
        else:
            ok("  RPM Fusion free already enabled")

        if _ffmpeg_free_installed():
            info("  Swapping ffmpeg-free → ffmpeg (this adds libx264) …")
            run_cmd(["sudo", "dnf", "swap", "-y", "ffmpeg-free", "ffmpeg",
                     "--allowerasing"])
        else:
            info("  Installing ffmpeg from RPM Fusion …")
            run_cmd(["sudo", "dnf", "install", "-y", "ffmpeg"])
        return True
    except subprocess.CalledProcessError as e:
        err(f"  Fedora ffmpeg install failed: {e}")
        warn("  Try manually: sudo dnf swap -y ffmpeg-free ffmpeg --allowerasing")
        return False

def _install_ffmpeg_debian(sudo):
    if not sudo:
        warn("Need sudo. Run: sudo apt-get install -y ffmpeg")
        return False
    try:
        run_cmd(["sudo", "apt-get", "update", "-qq"])
        run_cmd(["sudo", "apt-get", "install", "-y", "ffmpeg"])
        return True
    except subprocess.CalledProcessError as e:
        err(f"  apt-get install ffmpeg failed: {e}")
        return False

def _install_ffmpeg_arch(sudo):
    if not sudo:
        warn("Need sudo. Run: sudo pacman -S --noconfirm ffmpeg")
        return False
    try:
        run_cmd(["sudo", "pacman", "-S", "--noconfirm", "ffmpeg"])
        return True
    except subprocess.CalledProcessError as e:
        err(f"  pacman install ffmpeg failed: {e}")
        return False

def _install_ffmpeg_opensuse(sudo):
    if not sudo:
        warn("Need sudo. Run: sudo zypper install -y ffmpeg")
        return False
    try:
        run_cmd(["sudo", "zypper", "--non-interactive", "install", "ffmpeg"])
        return True
    except subprocess.CalledProcessError as e:
        err(f"  zypper install ffmpeg failed: {e}")
        return False

def _install_ffmpeg_mac():
    if not have("brew"):
        warn("Homebrew not found — install from https://brew.sh/ then: brew install ffmpeg")
        return False
    try:
        run_cmd(["brew", "install", "ffmpeg"])
        return True
    except subprocess.CalledProcessError as e:
        err(f"  brew install ffmpeg failed: {e}")
        return False

def _install_ffmpeg_windows():
    for mgr, cmd in [
        ("winget",  ["winget", "install", "--id", "Gyan.FFmpeg", "-e",
                     "--source", "winget", "--silent"]),
        ("choco",   ["choco", "install", "ffmpeg", "-y"]),
        ("scoop",   ["scoop", "install", "ffmpeg"]),
    ]:
        if have(mgr):
            try:
                run_cmd(cmd)
                return True
            except subprocess.CalledProcessError:
                warn(f"  {mgr} install failed, trying next …")
    warn("Could not auto-install ffmpeg on Windows.")
    warn("Download manually from: https://www.gyan.dev/ffmpeg/builds/")
    warn("  → ffmpeg-release-essentials.zip → extract → add bin/ to PATH")
    return False

def ensure_ffmpeg(os_type, distro, try_install, sudo):
    if _ffmpeg_ok():
        ok("ffmpeg + libx264 ready")
        return

    if have("ffmpeg") and not _has_libx264():
        warn("ffmpeg found but libx264 is MISSING — subtitle encoding will fail")
        if not try_install:
            if os_type in ("linux", "wsl") and distro == "fedora":
                warn("  Fix: python3 run.py --install-ffmpeg")
                warn("    or: sudo dnf swap -y ffmpeg-free ffmpeg --allowerasing")
            else:
                warn("  Fix: python3 run.py --install-ffmpeg")
            return
    elif not have("ffmpeg"):
        warn("ffmpeg not found — REQUIRED for video processing")
        if not try_install:
            warn("  Fix: python3 run.py --install-ffmpeg")
            return

    info("Installing / upgrading ffmpeg …")
    installed = False
    if os_type in ("linux", "wsl"):
        if distro == "fedora":
            installed = _install_ffmpeg_fedora(sudo)
        elif distro == "debian":
            installed = _install_ffmpeg_debian(sudo)
        elif distro == "arch":
            installed = _install_ffmpeg_arch(sudo)
        elif distro == "opensuse":
            installed = _install_ffmpeg_opensuse(sudo)
        elif distro == "rhel":
            installed = _install_ffmpeg_fedora(sudo)  # dnf-based
        else:
            installed = _install_ffmpeg_debian(sudo)   # best guess
    elif os_type == "mac":
        installed = _install_ffmpeg_mac()
    elif os_type == "windows":
        installed = _install_ffmpeg_windows()

    if installed:
        if _ffmpeg_ok():
            ok("ffmpeg + libx264 confirmed working")
        elif have("ffmpeg"):
            warn("ffmpeg installed but libx264 check failed — encoding may still work")
        else:
            warn("ffmpeg install reported success but binary not found — restart shell?")
    # else: distro-specific function already printed the warning

# ── Fonts ──────────────────────────────────────────────────────────────────────
def ensure_fonts(skip):
    if skip:
        ok("Skipping font downloads  (--skip-fonts)")
        return

    FONTS_DIR.mkdir(exist_ok=True)
    missing = [n for n in FONT_URLS if not (FONTS_DIR / n).exists()]

    if not missing:
        ok(f"All {len(FONT_URLS)} fonts already present in ./fonts/")
        return

    already = len(FONT_URLS) - len(missing)
    if already:
        info(f"{already} fonts already cached; downloading {len(missing)} missing …")
    else:
        info(f"Downloading {len(FONT_URLS)} fonts into ./fonts/ …")

    ok_count = fail_count = 0
    for name in missing:
        url   = FONT_URLS[name]
        dest  = FONTS_DIR / name
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = resp.read()
            dest.write_bytes(data)
            kb = len(data) // 1024
            if kb < 10:
                warn(f"  tiny ({kb} KB) — possibly corrupt, removing: {name}")
                dest.unlink(missing_ok=True)
                fail_count += 1
            else:
                print(f"  {c('✓', '32')}  {name}  ({kb} KB)")
                ok_count += 1
        except Exception as exc:
            warn(f"  FAILED  {name}  ({exc})")
            if dest.exists():
                dest.unlink(missing_ok=True)
            fail_count += 1

    if ok_count:
        ok(f"{ok_count} font(s) downloaded to ./fonts/")
    if fail_count:
        warn(f"{fail_count} font(s) failed — subtitles will fall back to whichever fonts are present")

# ── Templates ──────────────────────────────────────────────────────────────────
def ensure_templates():
    tdir   = HERE / "templates"
    tindex = tdir / "index.html"
    src    = HERE / "index.html"
    if tindex.exists():
        ok("templates/index.html in place")
        return
    if src.exists():
        tdir.mkdir(exist_ok=True)
        shutil.copy2(src, tindex)
        ok("Copied index.html → templates/index.html")
    else:
        warn("index.html not found — the web UI may not load")

# ── .env ───────────────────────────────────────────────────────────────────────
def ensure_env():
    env = HERE / ".env"
    if not env.exists():
        env.write_text(
            "# ShortsAI — fill in your API keys before processing videos\n"
            "GROQ_API_KEY=\n"
            "DEEPGRAM_API_KEY=\n"
            "# Required only for 'Best clips only' mode (Gemini).\n"
            "# Get one at https://aistudio.google.com/apikey\n"
            "GEMINI_API_KEY=\n"
        )
        warn(".env created — add your GROQ_API_KEY before processing videos")
        return

    lines = env.read_text().splitlines()

    def _is_set(name):
        return any(ln.startswith(f"{name}=") and ln.strip() != f"{name}=" for ln in lines)

    if _is_set("GROQ_API_KEY"):
        ok(".env — GROQ_API_KEY set")
    else:
        warn(".env found but GROQ_API_KEY is empty — add it before processing videos")

    # Gemini is optional: without it the "Best clips only" toggle falls back to
    # the standard Groq selector rather than failing, so this is a hint, not an error.
    if _is_set("GEMINI_API_KEY"):
        ok(".env — GEMINI_API_KEY set  ('Best clips only' mode enabled)")
    else:
        if not any(ln.startswith("GEMINI_API_KEY=") for ln in lines):
            with env.open("a") as f:
                f.write("\n# Required only for 'Best clips only' mode (Gemini).\n"
                        "# Get one at https://aistudio.google.com/apikey\n"
                        "GEMINI_API_KEY=\n")
            info(".env — added a GEMINI_API_KEY placeholder")
        warn(".env — GEMINI_API_KEY empty  "
             "('Best clips only' mode will fall back to the standard selector)")

# ── Self-test ─────────────────────────────────────────────────────────────────
# Each _*_TEST_CODE snippet runs inside the venv so it can import installed packages.
# Exit code convention: 0 = pass, 1 = fail, 2 = not-configured/skipped.

_GROQ_TEST_CODE = """\
import os, sys
try:
    from dotenv import load_dotenv; load_dotenv()
except ImportError:
    pass
key = os.getenv("GROQ_API_KEY", "").strip()
if not key:
    print("NO_KEY"); sys.exit(2)
try:
    from groq import Groq
    models = Groq(api_key=key).models.list()
    whisper = [m.id for m in models.data if "whisper" in m.id.lower()]
    print("OK  ({} models, whisper: {})".format(
        len(models.data), ", ".join(whisper) if whisper else "none listed"))
except Exception as e:
    print("FAIL  {}".format(e)); sys.exit(1)
"""

_GEMINI_TEST_CODE = """\
import os, sys
try:
    from dotenv import load_dotenv; load_dotenv()
except ImportError:
    pass
key = (os.getenv("GEMINI_API_KEY", "") or os.getenv("GOOGLE_API_KEY", "")).strip()
if not key:
    print("NO_KEY"); sys.exit(2)
try:
    from google import genai
except ImportError:
    print("FAIL  google-genai not installed — run: python3 run.py --reinstall"); sys.exit(1)
try:
    # Bind the client to a name: models.list() returns a LAZY pager, and a
    # temporary client gets garbage-collected (closing its transport) before
    # the pager is iterated -> "Cannot send a request, as the client has been closed".
    client = genai.Client(api_key=key)
    models = [m.name for m in client.models.list()]
    pro = [m for m in models if "2.5-pro" in m]
    print("OK  ({} models, 2.5-pro: {})".format(
        len(models), "yes" if pro else "not listed"))
except Exception as e:
    print("FAIL  {}".format(e)); sys.exit(1)
"""

_DEEPGRAM_TEST_CODE = """\
import os, sys, urllib.request, urllib.error
try:
    from dotenv import load_dotenv; load_dotenv()
except ImportError:
    pass
key = os.getenv("DEEPGRAM_API_KEY", "").strip()
if not key:
    print("NO_KEY"); sys.exit(2)
try:
    req = urllib.request.Request(
        "https://api.deepgram.com/v1/projects",
        headers={"Authorization": "Token " + key}
    )
    with urllib.request.urlopen(req, timeout=10) as r:
        print("OK  (HTTP {})".format(r.status))
except urllib.error.HTTPError as e:
    if e.code == 401:
        print("FAIL  401 Unauthorized — key is invalid"); sys.exit(1)
    print("FAIL  HTTP {}".format(e.code)); sys.exit(1)
except Exception as e:
    print("FAIL  {}".format(e)); sys.exit(1)
"""

# Minimal valid ASS file that exercises subtitle rendering.
_ASS_TEST_CONTENT = """\
[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
WrapStyle: 2

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,80,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,1,0,0,0,100,100,0,0,1,3,0,2,60,60,80,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
Dialogue: 0,0:00:00.00,0:00:01.00,Default,,0,0,0,,{\\an2}ShortsAI subtitle test
"""


def _venv_run(code, timeout=20):
    """Run a Python snippet inside the venv.
    Returns (status, message): status True=pass, False=fail, None=not-configured."""
    vpy = str(venv_python())
    if not venv_python().exists():
        return None, "venv not ready — run without --skip-install first"
    try:
        result = subprocess.run(
            [vpy, "-c", code],
            capture_output=True, text=True,
            timeout=timeout, cwd=str(HERE),
        )
        stdout = result.stdout.strip()
        stderr = result.stderr.strip()
        if result.returncode == 2:          # not-configured / optional
            return None, stdout or "not set"
        if result.returncode == 0:
            return True, stdout or "OK"
        last_err = stderr.splitlines()[-1] if stderr else ""
        return False, stdout or last_err or "unknown error"
    except subprocess.TimeoutExpired:
        return False, f"timed out after {timeout}s — check your internet connection"
    except Exception as e:
        return False, str(e)


def self_test():
    """Live-test both API keys and the full subtitle encode pipeline."""
    import tempfile

    hdr("Self-test")

    # 1 ── Groq API ────────────────────────────────────────────────────────────
    info("Groq API key …")
    status, msg = _venv_run(_GROQ_TEST_CODE, timeout=15)
    if status is True:
        ok(f"  Groq  {msg}")
    elif status is None:
        warn("  Groq  GROQ_API_KEY not set in .env")
        warn("    → Get a free key at console.groq.com")
    else:
        err(f"  Groq  INVALID — {msg}")
        warn("    → Check GROQ_API_KEY in .env  |  get a key at console.groq.com")

    # 2 ── Deepgram API (optional) ─────────────────────────────────────────────
    info("Deepgram API key (optional) …")
    status, msg = _venv_run(_DEEPGRAM_TEST_CODE, timeout=15)
    if status is True:
        ok(f"  Deepgram  {msg}")
    elif status is None:
        ok("  Deepgram  not configured  "
           "(optional — Hindi subtitles still work via Groq Whisper)")
    else:
        warn(f"  Deepgram  INVALID — {msg}")
        warn("    → Check DEEPGRAM_API_KEY in .env  |  leave it blank to disable")

    # 2b ─ Gemini API (optional — "Best clips only" mode) ──────────────────────
    info("Gemini API key (optional — 'Best clips only' mode) …")
    status, msg = _venv_run(_GEMINI_TEST_CODE, timeout=20)
    if status is True:
        ok(f"  Gemini  {msg}")
    elif status is None:
        ok("  Gemini  not configured  "
           "(optional — 'Best clips only' falls back to the standard selector)")
    else:
        warn(f"  Gemini  INVALID — {msg}")
        warn("    → Check GEMINI_API_KEY in .env  |  get a key at aistudio.google.com/apikey")

    # 3 ── Subtitle engine ─────────────────────────────────────────────────────
    info("Subtitle engine (ffmpeg ASS burn + libx264 encode) …")
    if not have("ffmpeg"):
        warn("  Subtitle engine  skipped — ffmpeg not found")
        warn("    → Run with --install-ffmpeg to fix")
        return

    try:
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir   = Path(tmp)
            ass_file = tmpdir / "test.ass"
            out_file = tmpdir / "out.mp4"

            ass_file.write_text(_ASS_TEST_CONTENT, encoding="utf-8")

            result = subprocess.run(
                [
                    "ffmpeg", "-y", "-loglevel", "error",
                    # synthetic 1-second 1080×1920 black source — no real input needed
                    "-f", "lavfi", "-i",
                    "color=c=black:size=1080x1920:rate=30:duration=1",
                    "-vf", f"ass={ass_file}",            # burn the subtitle
                    "-c:v", "libx264", "-preset", "ultrafast", "-crf", "35",
                    "-t", "1", str(out_file),
                ],
                capture_output=True, text=True, timeout=30,
            )
            if result.returncode == 0 and out_file.exists() \
                    and out_file.stat().st_size > 2_000:
                size_kb = out_file.stat().st_size // 1024
                ok(f"  Subtitle engine  OK  ({size_kb} KB test clip produced)")
            else:
                lines = (result.stderr or "").strip().splitlines()
                msg = lines[-1] if lines else "no output produced"
                err(f"  Subtitle engine  FAILED — {msg}")
                warn("    → Run with --install-ffmpeg to reinstall ffmpeg with libx264")
    except subprocess.TimeoutExpired:
        warn("  Subtitle engine  timed out (ffmpeg took > 30 s — machine may be overloaded)")
    except Exception as e:
        err(f"  Subtitle engine  unexpected error — {e}")


# ── Final readiness summary ────────────────────────────────────────────────────
def readiness_summary():
    issues = []
    if not _ffmpeg_ok():
        if not have("ffmpeg"):
            issues.append("ffmpeg missing  →  run with --install-ffmpeg")
        else:
            issues.append("ffmpeg has no libx264  →  run with --install-ffmpeg")
    fonts_present = sum(1 for n in FONT_URLS if (FONTS_DIR / n).exists())
    if fonts_present == 0:
        issues.append("no fonts in ./fonts/  →  run with --skip-fonts=false or bash fetch_fonts.sh")
    groq_set = False
    env = HERE / ".env"
    if env.exists():
        groq_set = any(
            ln.startswith("GROQ_API_KEY=") and ln.strip() != "GROQ_API_KEY="
            for ln in env.read_text().splitlines()
        )
    if not groq_set:
        issues.append("GROQ_API_KEY not set in .env  →  required for transcription")

    if issues:
        print()
        warn("Action needed before the app will fully work:")
        for issue in issues:
            print(f"  {c('→', '33')} {issue}")
    else:
        print()
        ok("All systems go — ready to process videos")

# ── Launch ─────────────────────────────────────────────────────────────────────
def launch(host, port, reload_):
    vpy = str(venv_python())
    cmd = [vpy, "-m", "uvicorn", "app:app", "--host", host, "--port", str(port)]
    if reload_:
        cmd.append("--reload")
    shown = host if host != "0.0.0.0" else "localhost"
    print()
    ok(f"Starting ShortsAI  →  http://{shown}:{port}")
    info("Press Ctrl+C to stop.\n")
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print()
        info("Stopped.")

# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description="Set up and run ShortsAI with one command.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--host",           default="127.0.0.1",
                    help="Bind host (use 0.0.0.0 to expose on LAN/phone)")
    ap.add_argument("--port",           type=int, default=8000,
                    help="Port (default 8000)")
    ap.add_argument("--reload",         action="store_true",
                    help="Dev mode: auto-restart when you edit Python files")
    ap.add_argument("--reinstall",      action="store_true",
                    help="Force re-run pip install even if already done")
    ap.add_argument("--skip-install",   action="store_true",
                    help="Skip pip install entirely (use if deps are already installed)")
    ap.add_argument("--install-ffmpeg", action="store_true",
                    help="Auto-install or fix ffmpeg via system package manager")
    ap.add_argument("--skip-fonts",     action="store_true",
                    help="Skip font downloads (faster re-runs after first setup)")
    ap.add_argument("--no-sudo",        action="store_true",
                    help="Skip every step that requires sudo")
    ap.add_argument("--setup-only",     action="store_true",
                    help="Run all setup steps but do not start the server")
    ap.add_argument("--skip-test",      action="store_true",
                    help="Skip API key and subtitle engine self-tests")
    args = ap.parse_args()

    os.chdir(HERE)
    os_type, distro = detect_os()
    sudo = not args.no_sudo

    label = f"{os_type}" + (f" / {distro}" if distro else "")
    print(c("\n  ShortsAI launcher", "1") + c(f"  ({label})\n", "90"))

    hdr("Python")
    check_python()
    fresh = ensure_venv()

    hdr("Python packages")
    if not args.skip_install:
        install_deps(force=args.reinstall or (fresh and not STAMP.exists()))
    else:
        warn("Skipping pip install  (--skip-install)")
    check_ytdlp()

    hdr("ffmpeg")
    ensure_ffmpeg(os_type, distro, args.install_ffmpeg, sudo)

    hdr("Fonts  (18 families)")
    ensure_fonts(args.skip_fonts)

    hdr("App files")
    ensure_templates()
    ensure_env()

    readiness_summary()

    if not args.skip_test:
        self_test()

    if args.setup_only:
        print()
        ok("Setup complete.  Start any time:  " + c("python3 run.py", "1"))
        return

    launch(args.host, args.port, args.reload)


if __name__ == "__main__":
    main()
