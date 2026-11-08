#!/usr/bin/env python3
"""
streamlit_app.py — Shorts Maker (Streamlit Community Cloud version).

Wraps pipeline.py: give it a long video (file upload OR YouTube URL) and it
returns up to 3 vertical 720x1280 clips cut from the most speech-dense
moments, with word-timed captions burned in.

This is the free-tier build (Hugging Face Gradio Spaces went PRO-only, so
the engine moved to Streamlit Community Cloud — ~1 GB RAM), therefore:
  - the Whisper model defaults to "tiny" ("base" is selectable but slower
    and heavier — it can run out of memory on the free tier),
  - source longer than ~20 minutes  -> rejected with a plain message,
  - upload larger than ~300 MB      -> rejected with a plain message,
  - every failure is caught and shown with st.error as a readable message;
    the app never shows a raw traceback.

The whole UI lives in main() and only runs under Streamlit / __main__, so
importing this module (e.g. for a syntax/import test) does nothing.
"""

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# Reuse the Phase-0-proven pipeline helpers (same folder in the repo).
from pipeline import (build_captions, ffprobe_duration, pick_windows, run,
                      write_ass)

MAX_DURATION_S = 20 * 60            # ~20 minutes (free-tier limit)
MAX_FILE_BYTES = 300 * 1024 * 1024  # ~300 MB (free-tier limit)
MAX_UPLOAD_MB = 300                 # st.file_uploader hard widget cap


def fmt_time(t):
    m, s = divmod(int(round(t)), 60)
    return f"{m}:{s:02d}"


def download_youtube(url, dest_dir):
    """Download a YouTube video with yt-dlp. Returns the local file Path.
    Raises RuntimeError with a readable message on failure."""
    out_tmpl = str(Path(dest_dir) / "source.%(ext)s")
    cmd = [sys.executable, "-m", "yt_dlp",
           "-f", "bv*[height<=720]+ba/b[height<=720]/b",
           "--merge-output-format", "mp4",
           "--max-filesize", "300M",
           "--no-playlist",
           "-o", out_tmpl]
    # yt-dlp needs a JS runtime for full YouTube extraction; the
    # Streamlit Cloud image has none by default (only deno is enabled,
    # and it is not installed), which made downloads fail with
    # "No supported JavaScript runtime" + HTTP 403. packages.txt
    # installs nodejs — point yt-dlp at it when it is present.
    # (CLI equivalent of the Python-API option
    # 'js_runtimes': {'node': {}}.)
    if shutil.which("node"):
        cmd += ["--js-runtimes", "node"]
    cmd.append(url)
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=1200)
    candidates = sorted(Path(dest_dir).glob("source.*"))
    candidates = [c for c in candidates if c.suffix != ".part"]
    if r.returncode != 0 or not candidates:
        tail = (r.stderr or r.stdout or "")[-400:].strip()
        raise RuntimeError(
            "Could not download that YouTube link. YouTube may block "
            "downloads from server addresses — please upload the video "
            "file instead. (Details: " + tail + ")")
    return candidates[0]


def make_clips(source: Path, model_name: str, workdir: Path):
    """Run the pipeline on a local source file inside workdir.
    Returns (clip_paths list of Path, status_text)."""
    outdir = workdir / "output"
    outdir.mkdir(parents=True, exist_ok=True)

    duration = ffprobe_duration(source)
    if duration > MAX_DURATION_S:
        raise ValueError(
            f"That video is {fmt_time(duration)} long. The free tool "
            "accepts videos up to about 20 minutes — please trim it and "
            "try again.")
    size = source.stat().st_size
    if size > MAX_FILE_BYTES:
        raise ValueError(
            f"That file is {size / 1e6:.0f} MB. The free tool accepts "
            "files up to about 300 MB — please compress it and try "
            "again.")

    # 1. audio
    wav = outdir / "audio.wav"
    run(["ffmpeg", "-y", "-v", "error", "-i", str(source), "-vn",
         "-ac", "1", "-ar", "16000", str(wav)])

    # 2. transcribe (word timestamps, VAD — same settings as Phase 0;
    #    model loaded fresh per run so its RAM is freed afterwards)
    from faster_whisper import WhisperModel
    model = WhisperModel(model_name, device="cpu", compute_type="int8")
    seg_iter, info = model.transcribe(str(wav), word_timestamps=True,
                                      vad_filter=True, language="en")
    segments = []
    for s in seg_iter:
        segments.append({"start": s.start, "end": s.end,
                         "text": s.text.strip(),
                         "words": [{"start": w.start, "end": w.end,
                                    "word": w.word}
                                   for w in (s.words or [])]})
    del model  # free the RAM before the ffmpeg renders start
    all_words = [w for s in segments for w in s["words"]]
    if not all_words:
        return [], ("No speech was found in that video, so no captioned "
                    "clips could be made. Try a video with clear spoken "
                    "audio.")

    # 3. pick best moments (rule-based v1 picker from pipeline.py)
    windows = pick_windows(segments, duration, 3)
    if not windows:
        return [], ("No suitable moments were found — the speech in that "
                    "video is too sparse or too broken up to cut clean "
                    "30-second clips. Try another video.")

    # 4. cut + vertical reframe (scale + blurred pad) + burn captions
    clip_paths = []
    lines = []
    for idx, w in enumerate(windows, 1):
        cues = build_captions(all_words, w["start"], w["end"])
        ass = outdir / f"clip{idx}.ass"
        write_ass(ass, cues)
        out = outdir / f"clip{idx}.mp4"
        vf = ("split[a][b];"
              "[b]scale=720:1280,crop=720:1280,gblur=sigma=40[bg];"
              "[a]scale=720:-2[fg];"
              "[bg][fg]overlay=(W-w)/2:(H-h)/2,"
              f"subtitles=clip{idx}.ass")
        run(["ffmpeg", "-y", "-v", "error", "-ss", str(w["start"]),
             "-i", str(source), "-t", str(w["duration"]),
             "-vf", vf, "-c:v", "libx264", "-preset", "fast", "-crf", "20",
             "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart",
             str(out)], cwd=str(outdir))
        clip_paths.append(out)
        lines.append(
            f"Clip {idx}: {fmt_time(w['start'])} – {fmt_time(w['end'])} "
            f"of the original ({w['duration']:.0f} seconds)")

    status = ("Done — the tool picked these moments from your "
              f"{fmt_time(duration)} video:\n" +
              "\n".join(lines) +
              "\nDownload the clips and post them to Shorts, TikTok or "
              "Reels.")
    return clip_paths, status


def main():
    import streamlit as st

    st.set_page_config(page_title="Shorts Maker — Long Video to Shorts",
                       page_icon="🎬")
    st.title("Shorts Maker — Long Video to Shorts")
    st.write(
        "Turn a long video into up to **3 vertical Shorts** with "
        "captions — free, no watermark, no sign-up. Upload a video file "
        "**or** paste a YouTube link (up to ~20 minutes / ~300 MB). "
        "Processing takes a few minutes on the free server — please "
        "keep this tab open.")
    st.caption("Only use videos you own or have the rights to reuse.")

    uploaded = st.file_uploader(
        "Upload a video file",
        type=["mp4", "mov", "mkv", "webm", "m4v"],
        help="Up to about 300 MB and 20 minutes on the free tier.")
    youtube_url = st.text_input(
        "…or paste a YouTube link",
        placeholder="https://www.youtube.com/watch?v=…")
    model_name = st.selectbox(
        "Caption quality (Whisper model)",
        options=["tiny", "base"], index=0,
        help="tiny (default) is fast and light — right for the free "
             "server. base gives slightly better captions but is "
             "slower and heavier, and can run out of memory on the "
             "free tier.")

    if not st.button("Make my Shorts", type="primary"):
        return

    youtube_url = (youtube_url or "").strip()
    if uploaded is not None and youtube_url:
        st.error("Please give ONE source — either upload a file or "
                 "paste a YouTube link, not both.")
        return
    if uploaded is None and not youtube_url:
        st.error("Add a video first: upload a file, or paste a YouTube "
                 "link.")
        return
    if youtube_url and not youtube_url.startswith(("http://",
                                                     "https://")):
        st.error("That does not look like a link. Paste the full "
                 "YouTube URL, or upload the video file instead.")
        return

    # Per-run scratch dir. The PREVIOUS run's dir is deleted now (its
    # clips have been served by then); the CURRENT run's dir is kept in
    # session state so st.video / download buttons can keep serving the
    # files after this script pass ends, and is cleaned up next run.
    old_dir = st.session_state.pop("workdir", None)
    if old_dir:
        shutil.rmtree(old_dir, ignore_errors=True)
    workdir = Path(tempfile.mkdtemp(prefix="shorts-maker-"))
    st.session_state["workdir"] = str(workdir)
    try:
        if uploaded is not None:
            if uploaded.size > MAX_FILE_BYTES:
                st.error("That file is over the ~300 MB free limit. "
                         "Please compress it and try again.")
                return
            suffix = Path(uploaded.name).suffix or ".mp4"
            source = workdir / f"source{suffix}"
            source.write_bytes(uploaded.getbuffer())
        else:
            with st.spinner("Downloading your video from YouTube…"):
                source = download_youtube(youtube_url, workdir)

        with st.spinner("Transcribing, picking the best moments and "
                        "rendering your Shorts — this can take a few "
                        "minutes…"):
            clips, status = make_clips(source, model_name, workdir)

        if not clips:
            st.error(status)
            return
        st.success(status)
        for idx, clip in enumerate(clips, 1):
            st.subheader(f"Clip {idx}")
            st.video(str(clip))
            st.download_button(
                f"Download clip {idx}",
                data=clip.read_bytes(),
                file_name=f"shorts-maker-clip{idx}.mp4",
                mime="video/mp4",
                key=f"download-{idx}")

    except (ValueError, RuntimeError) as e:
        st.error(f"Sorry — {e}")
    except SystemExit as e:
        # pipeline.run() raises SystemExit when an ffmpeg/ffprobe
        # call fails
        st.error("Sorry — something went wrong while processing that "
                 f"video ({e}). Try a different file, or a shorter "
                 "video.")
    except Exception as e:  # last-resort guard: never a traceback page
        st.error(f"Sorry — an unexpected error stopped the job ({e}). "
                 "Try again, or try a different video.")
        shutil.rmtree(workdir, ignore_errors=True)
        st.session_state.pop("workdir", None)


if __name__ == "__main__":
    main()
