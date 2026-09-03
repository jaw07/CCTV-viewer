#!/usr/bin/env python3
"""
Resolve Taiwan CWA (Central Weather Administration) coastal live cameras.

CWA restarts every stream daily, minting a NEW YouTube video ID each time, so a
hard-coded list of IDs goes stale within a day. This resolves the *current* live
video IDs from the official channel and maps them onto camera metadata by title.

Usage:
    python scripts/cwa_coastal.py                 # table of live cameras
    python scripts/cwa_coastal.py --json out.json # feed records for the viewer
    python scripts/cwa_coastal.py --west          # west-facing (Taiwan Strait) only
    python scripts/cwa_coastal.py --frames DIR    # also grab a JPEG from each

Requires yt-dlp on PATH (plus ffmpeg for --frames).
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path


# Camera metadata - including the sourced coordinates and their provenance -
# lives with the backend so the two cannot drift apart. This script is just a
# CLI over the same table.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))
from youtube_frames import CWA_CAMERAS as CAMERAS, CWA_CHANNEL  # noqa: E402


def resolve_live():
    """Return [(video_id, title)] for streams currently live on the channel."""
    try:
        out = subprocess.run(
            ["yt-dlp", "--no-warnings", "--flat-playlist", "--print", "%(id)s|%(title)s", CWA_CHANNEL],
            capture_output=True, text=True, timeout=180,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        sys.exit(f"could not run yt-dlp: {exc}")
    rows = []
    for line in out.stdout.strip().splitlines():
        if "|" in line:
            vid, _, title = line.partition("|")
            rows.append((vid.strip(), title.strip()))
    return rows


def match(title):
    """Map a stream title onto a camera entry by longest matching prefix."""
    for key in sorted(CAMERAS, key=len, reverse=True):
        if title.startswith(key):
            return key, CAMERAS[key]
    return None, None


def grab_frame(video_id, path):
    """Pull one JPEG from the live stream. Returns True on success."""
    url = subprocess.run(
        ["yt-dlp", "--no-warnings", "-f", "bv*[height<=720]/bv*/best[height<=720]/best", "-g",
         f"https://www.youtube.com/watch?v={video_id}"],
        capture_output=True, text=True, timeout=120,
    ).stdout.strip().splitlines()
    if not url:
        return False
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", url[0], "-frames:v", "1", "-q:v", "3", path],
        capture_output=True, timeout=120,
    )
    import os
    return os.path.exists(path) and os.path.getsize(path) > 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", metavar="FILE", help="write feed records as JSON")
    ap.add_argument("--west", action="store_true", help="only west-coast (Taiwan Strait) cameras")
    ap.add_argument("--frames", metavar="DIR", help="also grab one JPEG per camera into DIR")
    args = ap.parse_args()

    feeds, unknown = [], []
    for vid, title in resolve_live():
        key, meta = match(title)
        if not meta:
            unknown.append((vid, title))
            continue
        if args.west and meta["coast"] != "west":
            continue
        feeds.append({
            "id": f"CWA-{meta['slug']}",
            "videoId": vid,
            "streamUrl": f"https://www.youtube.com/watch?v={vid}",
            "embedUrl": f"https://www.youtube.com/embed/{vid}",
            "description": f"{meta['en']} ({title})",
            "roadName": "CWA coastal",
            "locationMile": meta["water"],
            "lat": meta["lat"],
            "lon": meta["lon"],
            "direction": meta["coast"],
            "source": "CWA",
            "locationSource": meta.get("loc_src", "unknown"),
        })

    if args.frames:
        import os
        os.makedirs(args.frames, exist_ok=True)
        for f in feeds:
            ok = grab_frame(f["videoId"], os.path.join(args.frames, f"{f['videoId']}.jpg"))
            f["frameOk"] = ok

    if args.json:
        with open(args.json, "w") as fh:
            json.dump(feeds, fh, ensure_ascii=False, indent=2)
        print(f"wrote {len(feeds)} feeds -> {args.json}")
    else:
        print(f"{'video id':<13} {'coast':<10} {'camera'}")
        for f in feeds:
            frame = "" if "frameOk" not in f else ("  frame:OK" if f["frameOk"] else "  frame:FAIL")
            print(f"{f['videoId']:<13} {f['direction']:<10} {f['description']}{frame}")
        print(f"\n{len(feeds)} live camera(s)")
        if unknown:
            print(f"{len(unknown)} live stream(s) not in the metadata map:")
            for vid, title in unknown:
                print(f"  {vid}  {title}")


if __name__ == "__main__":
    main()
