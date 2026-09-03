"""
Pull still frames from YouTube live cameras so they can flow through the same
snapshot + detection pipeline as the HTTP JPEG feeds.

Two facts drive this design:

1. Taiwan's CWA restarts every coastal stream daily, minting a NEW video ID each
   time. A hard-coded ID list breaks within 24h, so IDs are re-resolved from the
   official channel on an interval.
2. The direct media URLs yt-dlp returns are signed and expire (typically hours),
   so they are cached with a TTL and re-resolved on expiry or on failure.

Feeds backed by this module carry `imageUrl` of the form `ytframe://<camera_id>`;
the fetch paths in main.py intercept that scheme and read the cached frame here
rather than making an HTTP request.
"""
import asyncio
import os
from datetime import datetime, timezone
import shutil
import subprocess
import time
from typing import Dict, List, Optional, Tuple

try:
    from .frame_timestamp import extract_timestamp
except ImportError:  # direct execution
    from frame_timestamp import extract_timestamp

URL_SCHEME = "ytframe://"

# Title prefix -> metadata for the CWA coastal network. Coordinates are
# APPROXIMATE: CWA does not publish mast positions, so these place the marker in
# the correct harbour/headland, not on the exact camera.
CWA_CAMERAS = {
    "基隆和平島": dict(slug="keelung-heping", en="Keelung Heping Island", coast="north", lat=25.1594, lon=121.7614, water="East China Sea"),
    "碧砂": dict(slug="keelung-bisha", en="Keelung Bisha Fishing Port", coast="north", lat=25.1478, lon=121.7847, water="harbour"),
    "龍洞": dict(slug="longdong", en="New Taipei Longdong", coast="northeast", lat=25.1103, lon=121.9222, water="Pacific"),
    "新北福隆": dict(slug="fulong", en="New Taipei Fulong", coast="northeast", lat=25.0208, lon=121.9442, water="Pacific"),
    "宜蘭外澳": dict(slug="yilan-waiao", en="Yilan Wai'ao", coast="east", lat=24.8756, lon=121.8447, water="Pacific"),
    "宜蘭蘇澳": dict(slug="yilan-suao", en="Yilan Suao Port", coast="east", lat=24.5936, lon=121.8672, water="harbour"),
    "臺東富岡漁港": dict(slug="taitung-fugang", en="Taitung Fugang Fishing Port", coast="east", lat=22.7936, lon=121.1897, water="harbour"),
    "臺南安平港": dict(slug="tainan-anping", en="Tainan Anping Port", coast="west", lat=23.0028, lon=120.1600, water="harbour"),
    "新竹": dict(slug="hsinchu-cga", en="Hsinchu Coast Guard (12th Sea Patrol)", coast="west", lat=24.8500, lon=120.9200, water="Taiwan Strait"),
}

CWA_CHANNEL = "https://www.youtube.com/@cwa-tw/streams"


def tools_available() -> Tuple[bool, str]:
    """Both yt-dlp and ffmpeg are required; report which is missing."""
    missing = [t for t in ("yt-dlp", "ffmpeg") if shutil.which(t) is None]
    return (not missing), ("" if not missing else "missing: " + ", ".join(missing))


def _run(cmd: List[str], timeout: int) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


class YouTubeFrameGrabber:
    """Keeps one recent JPEG per camera, refreshed on a background loop."""

    def __init__(self, channel: str = CWA_CHANNEL, cameras: Optional[Dict] = None,
                 frame_interval: float = 20.0, id_refresh_interval: float = 3600.0,
                 media_url_ttl: float = 3600.0, grab_timeout: float = 30.0,
                 stream_latency: float = 0.0, logger=None):
        self.channel = channel
        self.cameras = cameras if cameras is not None else CWA_CAMERAS
        self.frame_interval = frame_interval
        self.id_refresh_interval = id_refresh_interval
        self.media_url_ttl = media_url_ttl
        self.grab_timeout = grab_timeout
        # YouTube live delivery runs behind real time (measured ~100s on the CWA
        # feeds). Frame capture times are shifted back by this so broadcast
        # detections carry the observation time, not the download time. Verify
        # against the timestamp CWA burns into the image before trusting it.
        self.stream_latency = stream_latency
        self.logger = logger

        # camera_id -> {"videoId", "title", "meta"}
        self.resolved: Dict[str, dict] = {}
        # camera_id -> (media_url, resolved_at)
        self._media: Dict[str, Tuple[str, float]] = {}
        # camera_id -> (jpeg_bytes, capture_epoch, time_source)
        self._frames: Dict[str, Tuple[bytes, float, str]] = {}
        self._last_id_refresh = 0.0

    # -- logging helper ----------------------------------------------------
    def _log(self, level: str, msg: str, **kw):
        if self.logger is None:
            return
        try:
            getattr(self.logger, level)(msg, **kw)
        except Exception:
            pass

    # -- id resolution -----------------------------------------------------
    def _match(self, title: str):
        for key in sorted(self.cameras, key=len, reverse=True):
            if title.startswith(key):
                return key, self.cameras[key]
        return None, None

    def refresh_ids(self) -> int:
        """Re-resolve which streams are live now. Returns camera count."""
        try:
            proc = _run(["yt-dlp", "--no-warnings", "--flat-playlist",
                         "--print", "%(id)s|%(title)s", self.channel], timeout=180)
        except Exception as exc:
            self._log("warning", "youtube id refresh failed", error=str(exc))
            return len(self.resolved)

        found = {}
        for line in proc.stdout.strip().splitlines():
            if "|" not in line:
                continue
            vid, _, title = line.partition("|")
            key, meta = self._match(title.strip())
            if not meta:
                continue
            found[meta["slug"]] = {"videoId": vid.strip(), "title": title.strip(), "meta": meta}

        if not found:
            # Keep the previous set rather than blanking the feed list on a blip.
            self._log("warning", "youtube id refresh returned nothing; keeping previous")
            return len(self.resolved)

        # Drop cached media URLs for cameras whose video ID changed.
        for key, entry in found.items():
            prev = self.resolved.get(key)
            if prev and prev["videoId"] != entry["videoId"]:
                self._media.pop(key, None)

        self.resolved = found
        self._last_id_refresh = time.time()
        self._log("info", "youtube ids refreshed", cameras=len(found))
        return len(found)

    # -- frame grabbing ----------------------------------------------------
    def _media_url(self, camera_id: str) -> Optional[str]:
        cached = self._media.get(camera_id)
        if cached and (time.time() - cached[1]) < self.media_url_ttl:
            return cached[0]

        entry = self.resolved.get(camera_id)
        if not entry:
            return None
        try:
            proc = _run(["yt-dlp", "--no-warnings", "-f", "best[height<=720]/best", "-g",
                         f"https://www.youtube.com/watch?v={entry['videoId']}"], timeout=120)
        except Exception:
            return None
        lines = [l for l in proc.stdout.strip().splitlines() if l.startswith("http")]
        if not lines:
            return None
        self._media[camera_id] = (lines[0], time.time())
        return lines[0]

    def grab(self, camera_id: str) -> bool:
        """Grab one frame. Retries once with a fresh media URL on failure."""
        for attempt in (0, 1):
            url = self._media_url(camera_id)
            if not url:
                return False
            try:
                proc = subprocess.run(
                    ["ffmpeg", "-y", "-loglevel", "error",
                     # Cap probing and socket waits so a stalled stream fails
                     # fast instead of pinning the loop for the whole timeout.
                     "-analyzeduration", "2M", "-probesize", "2M",
                     "-rw_timeout", "15000000",
                     "-i", url,
                     "-frames:v", "1", "-q:v", "3", "-f", "image2pipe", "-vcodec", "mjpeg", "-"],
                    capture_output=True, timeout=self.grab_timeout,
                )
            except Exception:
                proc = None
            if proc and proc.stdout and len(proc.stdout) > 1000:
                # Prefer the clock the camera burns into the frame - that is the
                # observation time. Download time only approximates it, and the
                # live stream runs ~100s behind, so it would misdate detections.
                stamped = extract_timestamp(proc.stdout)
                if stamped is not None:
                    capture_at, source = stamped, "image"
                else:
                    capture_at, source = time.time() - self.stream_latency, "download"
                self._frames[camera_id] = (proc.stdout, capture_at, source)
                self._log("info", "frame captured", camera=camera_id,
                          time_source=source,
                          captured_at=datetime.fromtimestamp(capture_at, timezone.utc)
                                              .strftime('%Y-%m-%dT%H:%M:%SZ'),
                          lag_seconds=round(time.time() - capture_at, 1))
                return True
            # Signed URL probably expired - drop it and resolve again.
            self._media.pop(camera_id, None)
        return False

    def get_frame(self, camera_id: str) -> Optional[bytes]:
        entry = self._frames.get(camera_id)
        return entry[0] if entry else None

    def frame_time_source(self, camera_id: str) -> Optional[str]:
        """'image' if read off the camera's own clock, else 'download'."""
        entry = self._frames.get(camera_id)
        return entry[2] if entry else None

    def frame_time(self, camera_id: str) -> Optional[float]:
        """Epoch seconds when this camera's cached frame was pulled."""
        entry = self._frames.get(camera_id)
        return entry[1] if entry else None

    def frame_age(self, camera_id: str) -> Optional[float]:
        entry = self._frames.get(camera_id)
        return (time.time() - entry[1]) if entry else None

    # -- feed records ------------------------------------------------------
    def build_feeds(self, west_only: bool = False) -> List[dict]:
        """Feed records shaped like the THB ones, for app_state.feeds_data."""
        feeds = []
        for camera_id, entry in self.resolved.items():
            meta = entry["meta"]
            if west_only and meta["coast"] != "west":
                continue
            feeds.append({
                "id": f"CWA-{camera_id}",  # camera_id is the ASCII slug
                "streamUrl": f"https://www.youtube.com/watch?v={entry['videoId']}",
                "imageUrl": f"{URL_SCHEME}{camera_id}",
                "description": f"{meta['en']} - {meta['water']}",
                "roadName": "CWA coastal",
                "locationMile": meta["water"],
                "lat": str(meta["lat"]),
                "lon": str(meta["lon"]),
                "direction": meta["coast"],
            })
        return feeds

    @staticmethod
    def camera_id_from_url(url: str) -> str:
        return url[len(URL_SCHEME):]

    # -- background loop ---------------------------------------------------
    async def run(self):
        """Refresh IDs periodically and keep one recent frame per camera.

        Frames are grabbed in a thread so ffmpeg never blocks the event loop.
        """
        loop = asyncio.get_event_loop()
        while True:
            try:
                if (time.time() - self._last_id_refresh) > self.id_refresh_interval:
                    await loop.run_in_executor(None, self.refresh_ids)

                cams = list(self.resolved)
                results = await asyncio.gather(
                    *(loop.run_in_executor(None, self.grab, c) for c in cams),
                    return_exceptions=True,
                )
                ok = sum(1 for r in results if r is True)
                self._log("info", "coastal frames grabbed", ok=ok, total=len(cams))
                for cam, r in zip(cams, results):
                    if r is not True:
                        self._log("debug", "frame grab failed", camera=cam,
                                  error=str(r) if isinstance(r, Exception) else None)

                await asyncio.sleep(self.frame_interval)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self._log("error", "youtube frame loop error", error=str(exc))
                await asyncio.sleep(self.frame_interval)
