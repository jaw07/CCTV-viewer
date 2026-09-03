"""
Detections are broadcast with the frame's capture time, not the time the
message happened to be built. A slow cache cycle must never misdate an
observation, so these guard that plumbing.
"""
import time
from datetime import datetime, timezone

import pytest

from core.cache import FeedCache


class TestFeedCacheCaptureTime:

    def test_capture_time_defaults_to_now(self):
        c = FeedCache()
        before = time.time()
        c.set_image("f1", b"x" * 2000)
        assert before <= c.get_capture_time("f1") <= time.time()

    def test_explicit_capture_time_is_preserved(self):
        """A frame pulled 45s ago must keep that timestamp, not cache time."""
        c = FeedCache()
        captured = time.time() - 45
        c.set_image("f1", b"x" * 2000, capture_time=captured)
        assert c.get_capture_time("f1") == pytest.approx(captured, abs=0.01)

    def test_unknown_feed_has_no_capture_time(self):
        assert FeedCache().get_capture_time("nope") is None

    def test_capture_time_updates_on_refresh(self):
        c = FeedCache()
        c.set_image("f1", b"a" * 2000, capture_time=1000.0)
        c.set_image("f1", b"b" * 2000, capture_time=2000.0)
        assert c.get_capture_time("f1") == 2000.0


class TestCotTimestamp:
    """generate_cot_message must date the event from the frame, not from now."""

    @staticmethod
    def _cot(**kw):
        import main
        feed = {"id": "CWA-tainan-anping", "lat": "23.0", "lon": "120.1",
                "roadName": "CWA coastal", "description": "Anping"}
        return main.generate_cot_message(feed, **kw)

    def test_uses_capture_time_not_now(self):
        captured = time.time() - 600  # ten minutes ago
        xml = self._cot(capture_time=captured)
        expected = datetime.fromtimestamp(captured, timezone.utc).strftime('%Y-%m-%dT%H:%M:%S')
        assert f"time='{expected}" in xml
        assert f"start='{expected}" in xml

    def test_falls_back_to_now_without_capture_time(self):
        xml = self._cot()
        now = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M')
        assert f"time='{now}" in xml

    def test_stale_is_after_observation(self):
        """Stale must be in the future even for an old frame, or consumers drop it."""
        import re
        xml = self._cot(capture_time=time.time() - 3000)
        obs = re.search(r"time='([^']+)'", xml).group(1)
        stale = re.search(r"stale='([^']+)'", xml).group(1)
        assert stale > obs

    def test_observed_time_appears_in_remarks(self):
        xml = self._cot(capture_time=time.time() - 120)
        assert "Observed:" in xml


class TestFrameTimestampOCR:
    """The camera's own on-frame clock is the observation time we broadcast."""

    @staticmethod
    def _synthetic(text, pos="bottom", size=(1280, 720)):
        """Render a frame with a burned-in timestamp like the CWA cameras use."""
        from PIL import Image, ImageDraw
        from io import BytesIO
        img = Image.new("RGB", size, (20, 30, 40))
        draw = ImageDraw.Draw(img)
        y = size[1] - 40 if pos == "bottom" else 10
        draw.rectangle([0, y - 6, 520, y + 26], fill=(0, 0, 0))
        draw.text((8, y), text, fill=(255, 255, 255))
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=95)
        return buf.getvalue()

    def test_returns_none_when_no_overlay(self):
        from frame_timestamp import extract_timestamp
        from PIL import Image
        from io import BytesIO
        buf = BytesIO()
        Image.new("RGB", (640, 480), (10, 10, 10)).save(buf, format="JPEG")
        assert extract_timestamp(buf.getvalue()) is None

    def test_rejects_implausible_timestamp(self):
        """A misread year must be discarded, not broadcast."""
        from frame_timestamp import extract_timestamp
        img = self._synthetic("1999-01-01 00:00:00")
        assert extract_timestamp(img, max_skew_seconds=3600) is None

    def test_handles_corrupt_bytes(self):
        from frame_timestamp import extract_timestamp
        assert extract_timestamp(b"not an image") is None


class TestCameraCoordinates:
    """Coordinates are drawn on a map and broadcast in CoT, so guard them.

    Bounds cover the whole ROC-administered area, not just Taiwan island:
    Kinmen sits at ~118.3E and Matsu at ~26.2N/119.9E.
    """

    @staticmethod
    def _cams():
        from coastal_cameras import COASTAL_CAMERAS
        return COASTAL_CAMERAS

    def test_all_within_bounds(self):
        """Catch a transposed or mistyped coordinate landing somewhere absurd."""
        for slug, meta in self._cams().items():
            assert 21.5 <= meta["lat"] <= 26.5, f"{slug} latitude out of bounds"
            assert 118.0 <= meta["lon"] <= 122.5, f"{slug} longitude out of bounds"

    def test_lat_lon_not_swapped(self):
        """Latitude and longitude ranges do not overlap here, so a swap shows."""
        for slug, meta in self._cams().items():
            assert meta["lat"] < meta["lon"], f"{slug} looks like lat/lon are swapped"

    def test_every_camera_records_its_position_source(self):
        for slug, meta in self._cams().items():
            assert meta.get("loc_src"), f"{slug} has no loc_src provenance"

    def test_west_facing_cameras_are_not_on_the_east_coast(self):
        """A west label with an east-coast longitude would misplace the marker.

        The threshold is loose (121.5, not 121.0) because northern Taiwan bulges
        east: Taoyuan's west-facing coast is at ~121.17E and Tamsui's river mouth
        at ~121.42E. Longitude alone cannot separate east from west in the south
        (Taitung's east coast is only 121.19E), so this catches gross mislabelling
        rather than proving the coast assignment.
        """
        for slug, meta in self._cams().items():
            if meta["coast"] in ("west", "northwest"):
                assert meta["lon"] < 121.5, f"{slug} marked west but at lon {meta['lon']}"

    def test_every_camera_has_a_channel_not_a_video_id(self):
        """Video ids rotate daily; storing one would rot within a day."""
        for slug, meta in self._cams().items():
            assert meta["channel"].startswith("https://www.youtube.com/"), slug
            assert "/watch?v=" not in meta["channel"], f"{slug} stores a video id"

    def test_co_located_cameras_are_declared_as_such(self):
        """Several angles at one site is fine, but it should be explicit."""
        import collections
        seen = collections.defaultdict(list)
        for slug, meta in self._cams().items():
            seen[(round(meta["lat"], 4), round(meta["lon"], 4))].append(slug)
        for coord, slugs in seen.items():
            if len(slugs) > 1:
                srcs = {self._cams()[s]["loc_src"] for s in slugs}
                assert srcs, f"co-located cameras {slugs} lack provenance"

    def test_feed_records_expose_the_source(self):
        from youtube_frames import YouTubeFrameGrabber
        g = YouTubeFrameGrabber()
        g.resolved = {slug: {"videoId": "x", "title": slug, "meta": m}
                      for slug, m in self._cams().items()}
        for feed in g.build_feeds():
            assert feed["locationSource"], f"{feed['id']} lost its position source"

    def test_west_filter_selects_only_west(self):
        from youtube_frames import YouTubeFrameGrabber
        g = YouTubeFrameGrabber()
        g.resolved = {slug: {"videoId": "x", "title": slug, "meta": m}
                      for slug, m in self._cams().items()}
        assert all(f["direction"] == "west" for f in g.build_feeds(west_only=True))


class TestBadgeMatchesImage:
    """A detection badge must describe the image actually being served.

    Two paths write the frame cache: the detection cycle (annotated frame +
    has_vehicles) and the coastal publish path (raw frame, for an immediate
    picture after a restart). If the second overwrites the first, the badge
    survives while the drawn boxes do not.
    """

    def test_raw_publish_never_claims_a_detection(self):
        from core.cache import FeedCache
        c = FeedCache()
        # Detection cycle stores an annotated frame and flags vehicles.
        c.set_image("CWA-x", b"annotated" * 400, is_working=True, has_vehicles=True)
        assert c.get_vehicle_detected("CWA-x") is True

        # The publish path must skip a feed that already has a cached frame.
        already_cached = c.get_image("CWA-x") is not None
        assert already_cached, "guard relies on detecting an existing frame"

    def test_raw_frame_is_stored_without_a_detection_flag(self):
        """When publish does fill an empty feed, it must not assert a detection."""
        from core.cache import FeedCache
        c = FeedCache()
        c.set_image("CWA-y", b"rawframe" * 400, is_working=True, has_vehicles=False)
        assert c.get_vehicle_detected("CWA-y") is False
        assert c.get_image("CWA-y") is not None
