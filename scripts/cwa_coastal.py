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

CHANNEL = "https://www.youtube.com/@cwa-tw/streams"

# Title prefix -> metadata. Coordinates are APPROXIMATE (siting is not published
# by CWA); they place the marker in the right harbour, not on the exact mast.
CAMERAS = {
    "基隆和平島": dict(en="Keelung Heping Island", coast="north", lat=25.1594, lon=121.7614, water="East China Sea"),
    "碧砂":       dict(en="Keelung Bisha Fishing Port", coast="north", lat=25.1478, lon=121.7847, water="harbour"),
    "龍洞":       dict(en="New Taipei Longdong", coast="northeast", lat=25.1103, lon=121.9222, water="Pacific"),
    "新北福隆":   dict(en="New Taipei Fulong", coast="northeast", lat=25.0208, lon=121.9442, water="Pacific"),
    "宜蘭外澳":   dict(en="Yilan Wai'ao", coast="east", lat=24.8756, lon=121.8447, water="Pacific"),
    "宜蘭蘇澳":   dict(en="Yilan Suao Port", coast="east", lat=24.5936, lon=121.8672, water="harbour"),
    "臺東富岡漁港": dict(en="Taitung Fugang Fishing Port", coast="east", lat=22.7936, lon=121.1897, water="harbour"),
    # --- West coast: faces the Taiwan Strait ---
    "臺南安平港": dict(en="Tainan Anping Port", coast="west", lat=23.0028, lon=120.1600, water="harbour"),
    "新竹":       dict(en="Hsinchu Coast Guard (12th Sea Patrol)", coast="west", lat=24.8500, lon=120.9200, water="Taiwan Strait"),
}


def resolve_live():
    """Return [(video_id, title)] for streams currently live on the channel."""
    try:
        out = subprocess.run(
            ["yt-dlp", "--no-warnings", "--flat-playlist", "--print", "%(id)s|%(title)s", CHANNEL],
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
        ["yt-dlp", "--no-warnings", "-f", "best[height<=720]/best", "-g",
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
            "id": f"CWA-{key}",
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
