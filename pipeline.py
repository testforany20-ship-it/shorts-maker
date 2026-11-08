#!/usr/bin/env python3
"""
pipeline.py — Long Video to Shorts, Phase 0 test pipeline (local only).

Usage:
    python pipeline.py <input_video> [--outdir DIR] [--model base|tiny] [--max-clips 3]

Pipeline:
  1. Extract mono 16 kHz audio with ffmpeg.
  2. Transcribe with faster-whisper (CPU, int8), word-level timestamps + VAD.
  3. Pick up to 3 non-overlapping "best moment" windows (30-60 s, aligned to
     whisper segment boundaries) scored by speech density / word rate /
     internal-silence penalty.
  4. For each window: cut with ffmpeg, reframe to vertical 720x1280 using
     SCALE + BLURRED PAD (foreground scaled to fit width, centred over a
     blurred, cropped copy of the same frame — the full source picture stays
     visible; chosen over centre-crop because the test source is 4:3 and a
     centre crop would discard most of the frame), burn word-timed captions
     from an .ass file generated from the whisper word timestamps.
  5. Write summary.json + summary.txt with chosen windows and first captions.

Outputs: clip1.mp4, clip2.mp4, clip3.mp4 (as many as windows found).
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

MIN_WIN = 30.0   # seconds
MAX_WIN = 60.0   # seconds


def run(cmd, **kw):
    print("+", " ".join(str(c) for c in cmd), flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True, **kw)
    if r.returncode != 0:
        print(r.stdout[-2000:])
        print(r.stderr[-3000:])
        raise SystemExit(f"command failed ({r.returncode}): {cmd[0]}")
    return r


def ffprobe_duration(path):
    r = run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "json", str(path)])
    return float(json.loads(r.stdout)["format"]["duration"])


def ass_time(t):
    h = int(t // 3600); m = int((t % 3600) // 60); s = t % 60
    return f"{h}:{m:02d}:{s:05.2f}"


def build_captions(words, win_start, win_end):
    """Group window-relative words into caption cues of <=5 words."""
    rel = [(w["start"] - win_start, w["end"] - win_start, w["word"].strip())
           for w in words if w["end"] > win_start and w["start"] < win_end]
    rel = [w for w in rel if w[2]]
    cues, cur = [], []
    for w in rel:
        if cur and (len(cur) >= 5 or w[0] - cur[-1][1] > 0.8
                    or cur[-1][2].endswith((".", "?", "!"))):
            cues.append(cur); cur = []
        cur.append(w)
    if cur:
        cues.append(cur)
    out = []
    for i, cue in enumerate(cues):
        start = max(0.0, cue[0][0])
        end = cue[-1][1] + 0.35
        if i + 1 < len(cues):
            end = min(end, cues[i + 1][0][0])
        if end - start < 0.25:
            end = start + 0.25
        out.append((start, end, " ".join(w[2] for w in cue)))
    return out


ASS_HEADER = """[Script Info]
ScriptType: v4.00+
PlayResX: 720
PlayResY: 1280
WrapStyle: 2

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Cap,DejaVu Sans,54,&H00FFFFFF,&H000000FF,&H00000000,&H64000000,-1,0,0,0,100,100,0,0,1,4,1,2,40,40,150,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


def write_ass(path, cues):
    def esc(t):
        return t.replace("\\", "").replace("{", "").replace("}", "")
    lines = [ASS_HEADER]
    for s, e, txt in cues:
        lines.append(f"Dialogue: 0,{ass_time(s)},{ass_time(e)},Cap,,0,0,0,,{esc(txt)}")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def pick_windows(segments, duration, max_clips):
    """Candidate window = consecutive whole segments starting at a segment
    start, extended until >= MIN_WIN without exceeding MAX_WIN (the final
    segment may close the window slightly past MAX_WIN only if it starts
    before MAX_WIN is reached and ends within MAX_WIN + 8)."""
    cands = []
    for i, seg in enumerate(segments):
        start = seg["start"]; words = 0; speech = 0.0; end = start; j = i
        while j < len(segments):
            s = segments[j]
            new_end = s["end"]
            if new_end - start > MAX_WIN + 8 and j > i:
                break
            end = new_end
            words += len(s.get("words", []))
            speech += sum(w["end"] - w["start"] for w in s.get("words", []))
            j += 1
            if end - start >= MIN_WIN:
                break
        dur = end - start
        if dur < MIN_WIN * 0.8:      # tolerate a slightly short final window
            continue
        # internal silence: gaps between segments inside the window
        gaps, prev_end = 0.0, start
        for s in segments[i:j]:
            gaps += max(0.0, s["start"] - prev_end)
            prev_end = s["end"]
        speech_ratio = speech / dur if dur else 0
        wpm = words / dur * 60 if dur else 0
        score = speech_ratio * 2.0 + min(wpm, 220) / 220.0 - gaps / dur
        cands.append({"start": round(start, 2), "end": round(end, 2),
                      "duration": round(dur, 2), "words": words,
                      "speech_ratio": round(speech_ratio, 3),
                      "gap_s": round(gaps, 2), "score": round(score, 4)})
    cands.sort(key=lambda c: -c["score"])
    chosen = []
    for c in cands:
        if all(c["end"] <= k["start"] or c["start"] >= k["end"] for k in chosen):
            chosen.append(c)
        if len(chosen) >= max_clips:
            break
    return sorted(chosen, key=lambda c: c["start"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("--outdir", default="output")
    ap.add_argument("--model", default="base")
    ap.add_argument("--max-clips", type=int, default=3)
    args = ap.parse_args()

    t0 = time.time()
    inp = Path(args.input).resolve()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    timing = {}

    duration = ffprobe_duration(inp)
    print(f"input: {inp}  duration: {duration:.1f}s")

    # 1. audio
    t = time.time()
    wav = outdir / "audio.wav"
    run(["ffmpeg", "-y", "-v", "error", "-i", str(inp), "-vn", "-ac", "1",
         "-ar", "16000", str(wav)])
    timing["audio_extract_s"] = round(time.time() - t, 1)

    # 2. transcribe
    t = time.time()
    from faster_whisper import WhisperModel
    model = WhisperModel(args.model, device="cpu", compute_type="int8")
    timing["model_load_s"] = round(time.time() - t, 1)
    t = time.time()
    seg_iter, info = model.transcribe(str(wav), word_timestamps=True,
                                      vad_filter=True, language="en")
    segments = []
    for s in seg_iter:
        segments.append({"start": s.start, "end": s.end, "text": s.text.strip(),
                         "words": [{"start": w.start, "end": w.end,
                                    "word": w.word} for w in (s.words or [])]})
    timing["transcribe_s"] = round(time.time() - t, 1)
    all_words = [w for s in segments for w in s["words"]]
    print(f"language: {info.language} (p={info.language_probability:.2f})  "
          f"segments: {len(segments)}  words: {len(all_words)}")
    (outdir / "transcript.json").write_text(
        json.dumps({"language": info.language, "segments": segments}, indent=1))

    # 3. windows
    windows = pick_windows(segments, duration, args.max_clips)
    print(f"windows chosen: {len(windows)}")
    for w in windows:
        print("  ", w)

    # 4. clips
    t = time.time()
    results = []
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
             "-i", str(inp), "-t", str(w["duration"]),
             "-vf", vf, "-c:v", "libx264", "-preset", "fast", "-crf", "20",
             "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart",
             str(out)], cwd=str(outdir))
        results.append({**w, "file": str(out),
                        "size_bytes": out.stat().st_size,
                        "caption_cues": len(cues),
                        "first_captions": [c[2] for c in cues[:3]]})
    timing["clip_render_s"] = round(time.time() - t, 1)
    timing["total_s"] = round(time.time() - t0, 1)

    summary = {"input": str(inp), "input_duration_s": round(duration, 1),
               "model": args.model, "compute": "cpu/int8",
               "reframe": "scale + blurred pad to 720x1280",
               "timing": timing, "clips": results}
    (outdir / "summary.json").write_text(json.dumps(summary, indent=1))
    txt = [f"Input: {inp} ({duration:.0f}s)  Model: faster-whisper {args.model} (CPU int8)",
           f"Total processing: {timing['total_s']}s  (audio {timing['audio_extract_s']}s, "
           f"model load {timing['model_load_s']}s, transcribe {timing['transcribe_s']}s, "
           f"render {timing['clip_render_s']}s)", ""]
    for i, r in enumerate(results, 1):
        txt.append(f"clip{i}.mp4  {r['start']:.1f}s -> {r['end']:.1f}s "
                   f"({r['duration']:.0f}s, score {r['score']}, "
                   f"{r['size_bytes']/1e6:.1f} MB)")
        for c in r["first_captions"]:
            txt.append(f"    caption: {c}")
    (outdir / "summary.txt").write_text("\n".join(txt) + "\n")
    print("\n".join(txt))
    print(f"\nDONE in {timing['total_s']}s -> {outdir}")


if __name__ == "__main__":
    main()
