# Shorts Maker — Long Video to Shorts

A free web tool that turns a long video into up to **3 vertical Shorts**
(720×1280) with word-timed captions burned in — no watermark, no sign-up.
Give it a video file upload **or** a YouTube link; it transcribes the
speech (faster-whisper), picks the most speech-dense moments, reframes
them to vertical with a blurred-pad layout, and returns downloadable
MP4 clips ready for YouTube Shorts, TikTok and Reels.

## Files

| File | Purpose |
|---|---|
| `streamlit_app.py` | The Streamlit app (UI + orchestration). |
| `pipeline.py` | The Phase-0-proven pipeline helpers it reuses (window picking, caption generation, .ass writer, ffprobe/ffmpeg runners). |
| `requirements.txt` | Python dependencies. |
| `packages.txt` | System packages for Streamlit Cloud (`ffmpeg`). |

## Deployment (Streamlit Community Cloud)

This folder is the root of the GitHub repo
`testforany20-ship-it/shorts-maker`. Deploy it on
[share.streamlit.io](https://share.streamlit.io): New app → pick the
repo → main file `streamlit_app.py` → Deploy. No secrets are needed.

Free-tier limits built into the app: Whisper model **tiny** by default
(base selectable, heavier), sources up to ~20 minutes / ~300 MB.
Only use videos you own or have the rights to reuse.
