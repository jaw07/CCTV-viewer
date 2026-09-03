"""
Read the timestamp the camera burns into its own video frame.

For broadcast detections, what matters is when the scene was observed - not when
we downloaded it. YouTube live delivery runs well behind real time (measured
~100s on the CWA feeds), so download time misdates an observation. The CWA
cameras overlay their own clock on every frame, which is authoritative.

Overlay formats seen on the CWA network (all bottom strip, white on dark):
    安平港-1 2026-09-04 03:10:54
    2026-09-04-03:27:34 AM和平島-1-cam1
    2026-09-0403:11:29-新竹海巡署艦隊分署第十二海巡隊-cam-1

so the parser is deliberately loose about separators between the date and time.
Times are camera-local (Taiwan, UTC+8) unless a different tz offset is given.
"""
import re
from datetime import datetime, timedelta, timezone
from io import BytesIO
from typing import Optional

# Digits with optional separators: YYYY-MM-DD[ -]HH:MM:SS
_TS_RE = re.compile(
    r"(20\d{2})\D{0,2}(\d{2})\D{0,2}(\d{2})"   # date
    r"\D{0,3}"                                   # separator (space, dash, none)
    r"(\d{2})\D{0,2}(\d{2})\D{0,2}(\d{2})"      # time
)

# Fallback when the date half is lost against a bright background: the clock
# alone is enough, because a live frame is always within minutes of now.
_TIME_ONLY_RE = re.compile(r"(\d{2}):(\d{2}):(\d{2})")

DEFAULT_TZ = timezone(timedelta(hours=8))  # Taiwan


def _ocr(image, config: str) -> str:
    import pytesseract
    return pytesseract.image_to_string(image, config=config)


def extract_timestamp(img_bytes: bytes, tz: timezone = DEFAULT_TZ,
                      max_skew_seconds: float = 86400) -> Optional[float]:
    """Return epoch seconds from the frame's burned-in clock, or None.

    The overlay sits at the top on some CWA cameras and the bottom on others,
    and a few carry none at all, so both strips are tried before giving up.
    `max_skew_seconds` discards implausible reads (a misread year, say) so the
    caller falls back to download time rather than broadcasting a bad one.
    """
    try:
        from PIL import Image
    except ImportError:
        return None

    try:
        img = Image.open(BytesIO(img_bytes))
    except Exception:
        return None

    import time as _t
    width, height = img.size
    # Overlay placement varies by camera, and how much scene the crop includes
    # materially changes what OCR can read, so several bands are tried.
    strips = [
        img.crop((0, int(height * 0.92), width, height)),
        img.crop((0, int(height * 0.88), width, height)),
        img.crop((0, int(height * 0.84), width, height)),
        img.crop((0, 0, width, int(height * 0.10))),
        img.crop((0, 0, width, int(height * 0.14))),
    ]

    for strip in strips:
        grey = strip.convert("L")
        grey = grey.resize((grey.width * 3, grey.height * 3), Image.LANCZOS)
        # 180 first: the overlay is bright text, and a high cut drops most of
        # the scene behind it. Lower cuts help when the plate is dimmer.
        for threshold in (180, 200, 140, 100):
            binarised = grey.point(lambda p, t=threshold: 255 if p > t else 0)
            try:
                text = _ocr(binarised, "--psm 7 -c tessedit_char_whitelist=0123456789-:/ ")
            except Exception:
                return None
            match = _TS_RE.search(text.replace("\n", " ").replace(" ", ""))
            if not match:
                continue
            year, month, day, hour, minute, second = (int(g) for g in match.groups())
            try:
                stamp = datetime(year, month, day, hour, minute, second, tzinfo=tz)
            except ValueError:
                continue
            epoch = stamp.timestamp()
            if abs(_t.time() - epoch) <= max_skew_seconds:
                return epoch

    # Second pass: accept a bare HH:MM:SS and anchor it to whichever calendar
    # day puts it closest to now. A live frame is only minutes old, so the day
    # is unambiguous.
    for strip in strips:
        grey = strip.convert("L")
        grey = grey.resize((grey.width * 3, grey.height * 3), Image.LANCZOS)
        for threshold in (180, 200, 140):
            binarised = grey.point(lambda p, t=threshold: 255 if p > t else 0)
            try:
                text = _ocr(binarised, "--psm 7 -c tessedit_char_whitelist=0123456789-:/ ")
            except Exception:
                return None
            match = _TIME_ONLY_RE.search(text)
            if not match:
                continue
            hour, minute, second = (int(g) for g in match.groups())
            if hour > 23 or minute > 59 or second > 59:
                continue
            now_local = datetime.now(tz)
            best = None
            for day_shift in (-1, 0, 1):
                base = (now_local + timedelta(days=day_shift)).date()
                try:
                    cand = datetime(base.year, base.month, base.day,
                                    hour, minute, second, tzinfo=tz).timestamp()
                except ValueError:
                    continue
                if best is None or abs(_t.time() - cand) < abs(_t.time() - best):
                    best = cand
            if best is not None and abs(_t.time() - best) <= max_skew_seconds:
                return best
    return None
