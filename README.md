# windowsMicPaste — local push-to-talk voice typing for Windows

A tiny Windows system-tray app that turns speech into text **100% locally** — no
cloud, no API key, nothing leaves your machine. Press a global hotkey, talk, and
the transcription is pasted straight into whatever app has focus.

Powered by [faster-whisper](https://github.com/SYSTRAN/faster-whisper) on your CPU.

## Features
- **Push-to-talk** — `Ctrl+Shift+D` (Win32 global hotkey, no admin) or left-click the tray icon.
- **Fully offline** — the local `base.en` Whisper model does the transcription; no internet, no API keys, private by default.
- **Auto-paste** — result is copied to the clipboard and pasted (`Ctrl+V`) into the active window.
- **Spoken punctuation** — say "comma", "period", "new line", etc. and it's inserted as real punctuation, with sentence capitalization.
- **Tray feedback** — icon turns red while recording, plus start/stop/done sounds and Windows notifications.

## Setup
```
pip install -r requirements.txt
python whisper_tray.py
```
Then press **Ctrl+Shift+D** (or click the tray icon) to start/stop recording.

## Configuration
Edit the constants at the top of `whisper_tray.py`:
- `HOTKEY` — toggle hotkey (default `ctrl+shift+d`).
- `WHISPER_MODEL` — `tiny.en` / `base.en` / `small.en` / `medium.en` / `large-v3` (bigger = more accurate, slower).
- `WHISPER_DEVICE` / `WHISPER_COMPUTE` — set to `cuda` / `float16` for an NVIDIA GPU.

## License
MIT — see [LICENSE](LICENSE).
