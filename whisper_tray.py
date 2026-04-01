"""
Whisper Push-to-Talk - Windows 11 System Tray App
- Left-click tray icon OR press hotkey (Ctrl+Alt+Space) to toggle recording
- While recording, icon turns red + Windows notification pops up
- On stop, faster-whisper transcribes and dumps to clipboard
- Paste anywhere with Ctrl+V

Requirements:
    pip install faster-whisper sounddevice numpy pyperclip pystray pillow pywin32
"""

import threading
import time
import sys
import ctypes
import ctypes.wintypes
import winsound
import numpy as np
import sounddevice as sd
import pyperclip
from faster_whisper import WhisperModel
from PIL import Image, ImageDraw
import pystray
from pystray import MenuItem as item

# ── Config ────────────────────────────────────────────────────────────────────
HOTKEY = "ctrl+shift+d"           # Global hotkey to toggle recording
SAMPLE_RATE = 16000               # Required by Whisper
WHISPER_MODEL = "base.en"         # Options: tiny.en, base.en, small.en, medium.en, large-v3
WHISPER_DEVICE = "cpu"            # "cpu" or "cuda" if you have a GPU
WHISPER_COMPUTE = "int8"          # "int8" (fast/CPU) or "float16" (GPU)
# ─────────────────────────────────────────────────────────────────────────────

# Global state
is_recording = False
audio_frames = []
recording_lock = threading.Lock()
tray_icon = None
model = None
model_ready = False
status_text = "Loading model..."
stream = None


def beep_start():
    winsound.PlaySound(r"C:\Windows\Media\Speech On.wav", winsound.SND_FILENAME | winsound.SND_ASYNC)

def beep_stop():
    winsound.PlaySound(r"C:\Windows\Media\Speech Off.wav", winsound.SND_FILENAME | winsound.SND_ASYNC)

def beep_done():
    winsound.PlaySound(r"C:\Windows\Media\Windows Notify.wav", winsound.SND_FILENAME | winsound.SND_ASYNC)


def show_notification(title: str, msg: str):
    """Show a Windows 11 balloon notification from the tray icon."""
    if tray_icon is not None:
        try:
            tray_icon.notify(msg, title)
        except Exception:
            pass


def make_icon(recording: bool) -> Image.Image:
    """Draw a tray icon. Red = recording, grey = idle."""
    size = 64
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    if recording:
        draw.ellipse([4, 4, 60, 60], fill=(220, 40, 40, 255))
        draw.ellipse([18, 18, 46, 46], fill=(255, 255, 255, 200))
    else:
        draw.ellipse([4, 4, 60, 60], fill=(80, 80, 80, 220))
        draw.ellipse([18, 18, 46, 46], fill=(200, 200, 200, 200))

    return img


def update_tray():
    """Refresh the tray icon and tooltip."""
    if tray_icon is None:
        return
    tray_icon.icon = make_icon(is_recording)
    if is_recording:
        tray_icon.title = "RECORDING... (press hotkey to stop)"
    else:
        tray_icon.title = f"Whisper PTT — {status_text}"


def audio_callback(indata, frames, time_info, status):
    """Called by sounddevice for each audio chunk."""
    if is_recording:
        audio_frames.append(indata.copy())


def start_recording():
    global is_recording, audio_frames, stream, status_text
    with recording_lock:
        if is_recording:
            return
        is_recording = True
        audio_frames = []
        status_text = "Recording..."

    update_tray()
    beep_start()
    show_notification("Whisper PTT", "Recording started... Press hotkey again to stop.")

    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        callback=audio_callback,
        blocksize=1024,
    )
    stream.start()
    print("Recording started")


def stop_recording():
    global is_recording, stream, status_text

    with recording_lock:
        if not is_recording:
            return
        is_recording = False
        status_text = "Transcribing..."

    update_tray()
    beep_stop()

    try:
        stream.stop()
        stream.close()
    except Exception:
        pass

    if not audio_frames:
        status_text = "Ready"
        update_tray()
        return

    show_notification("Whisper PTT", "Transcribing...")
    threading.Thread(target=transcribe_and_copy, daemon=True).start()


import re

PUNCTUATION_MAP = {
    r"\bperiod\b": ".",
    r"\bfull stop\b": ".",
    r"\bcomma\b": ",",
    r"\bexclamation point\b": "!",
    r"\bexclamation mark\b": "!",
    r"\bquestion mark\b": "?",
    r"\bcolon\b": ":",
    r"\bsemicolon\b": ";",
    r"\bsemi colon\b": ";",
    r"\bellipsis\b": "...",
    r"\bdash\b": "—",
    r"\bhyphen\b": "-",
    r"\bopen quote\b": '"',
    r"\bclose quote\b": '"',
    r"\bopen paren\b": "(",
    r"\bclose paren\b": ")",
    r"\bnew line\b": "\n",
    r"\bnewline\b": "\n",
    r"\bnew paragraph\b": "\n\n",
}

def apply_punctuation_commands(text: str) -> str:
    """Replace spoken punctuation words with actual punctuation."""
    for pattern, replacement in PUNCTUATION_MAP.items():
        # Replace the word, removing extra space before punctuation
        text = re.sub(r"\s*" + pattern + r"\s*", replacement, text, flags=re.IGNORECASE)
        # If punctuation was placed, capitalize the next letter
        if replacement in ".!?\n":
            text = re.sub(
                re.escape(replacement) + r"(\s*)([a-z])",
                lambda m: replacement + m.group(1) + m.group(2).upper(),
                text,
            )
    return text


def transcribe_and_copy():
    global status_text

    if not model_ready:
        status_text = "Model not ready yet"
        update_tray()
        return

    print("Transcribing...")
    audio = np.concatenate(audio_frames, axis=0).flatten()

    segments, info = model.transcribe(
        audio,
        beam_size=5,
        language="en",
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=300),
    )

    text = " ".join(seg.text.strip() for seg in segments).strip()
    text = apply_punctuation_commands(text)

    if text:
        pyperclip.copy(text)
        status_text = f"Copied! ({len(text)} chars)"
        print(f"Copied to clipboard: {text[:80]}{'...' if len(text) > 80 else ''}")
        beep_done()
        show_notification("Whisper PTT — Copied!", text[:200])
        # Auto-paste with Ctrl+V
        time.sleep(0.15)
        import win32com.client
        shell = win32com.client.Dispatch("WScript.Shell")
        shell.SendKeys("^v")
    else:
        status_text = "Nothing detected"
        print("No speech detected")
        show_notification("Whisper PTT", "No speech detected")

    update_tray()

    time.sleep(4)
    status_text = "Ready — " + HOTKEY
    update_tray()


def toggle_recording():
    if is_recording:
        stop_recording()
    else:
        start_recording()


def load_model():
    """Load Whisper model in background so app starts fast."""
    global model, model_ready, status_text
    print(f"Loading Whisper model '{WHISPER_MODEL}'...")
    status_text = f"Loading {WHISPER_MODEL}..."
    update_tray()

    model = WhisperModel(
        WHISPER_MODEL,
        device=WHISPER_DEVICE,
        compute_type=WHISPER_COMPUTE,
    )
    model_ready = True
    status_text = "Ready — " + HOTKEY
    print(f"Model loaded. Press {HOTKEY} or click tray icon to record.")
    update_tray()
    show_notification("Whisper PTT", f"Model loaded. Press {HOTKEY} to record.")


def quit_app(icon, item):
    icon.stop()
    sys.exit(0)


# ── Win32 global hotkey (no admin required) ───────────────────────────────────
HOTKEY_ID = 1
MOD_CTRL = 0x0002
MOD_SHIFT = 0x0004
VK_D = 0x44


def hotkey_listener():
    """Register Ctrl+Shift+D as a global hotkey via Win32 API."""
    user32 = ctypes.windll.user32
    if not user32.RegisterHotKey(None, HOTKEY_ID, MOD_CTRL | MOD_SHIFT, VK_D):
        print("Failed to register hotkey — try a different combo or close conflicting apps")
        return
    print(f"Hotkey registered: {HOTKEY}")

    msg = ctypes.wintypes.MSG()
    while user32.GetMessageW(ctypes.byref(msg), None, 0, 0) != 0:
        if msg.message == 0x0312 and msg.wParam == HOTKEY_ID:  # WM_HOTKEY
            toggle_recording()

    user32.UnregisterHotKey(None, HOTKEY_ID)
# ──────────────────────────────────────────────────────────────────────────────


def main():
    global tray_icon

    menu = (
        item("Toggle Recording", lambda icon, item: toggle_recording()),
        item("─────────────", lambda: None, enabled=False),
        item(f"Hotkey: {HOTKEY}", lambda: None, enabled=False),
        item("Quit", quit_app),
    )

    tray_icon = pystray.Icon(
        "WhisperPTT",
        make_icon(False),
        title="Whisper PTT — Loading...",
        menu=menu,
    )

    # Load model in background
    threading.Thread(target=load_model, daemon=True).start()

    # Register global hotkey via Win32 API (no admin needed)
    threading.Thread(target=hotkey_listener, daemon=True).start()

    # Run tray (blocks until quit)
    tray_icon.run()


if __name__ == "__main__":
    main()
