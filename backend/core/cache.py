"""
LRU cache with size limits and TTL.
"""
import time
import hashlib
from collections import OrderedDict
from typing import Optional, Dict, Any, TypeVar, Generic
from dataclasses import dataclass, field
from threading import Lock
import asyncio

T = TypeVar('T')


@dataclass
class CacheEntry(Generic[T]):
    value: T
    size_bytes: int
    created_at: float
    last_accessed: float
    access_count: int = 0
    hash: Optional[str] = None


class LRUCache(Generic[T]):
    """Thread-safe LRU cache with max items, max size, and optional TTL."""

    def __init__(
        self,
        max_items: int = 10000,
        max_size_bytes: int = 1024 * 1024 * 1024,  # 1 GB default
        ttl_seconds: Optional[float] = None,
    ):
        self.max_items = max_items
        self.max_size_bytes = max_size_bytes
        self.ttl_seconds = ttl_seconds

        self._cache: OrderedDict[str, CacheEntry[T]] = OrderedDict()
        self._lock = Lock()
        self._total_size = 0

        # Statistics
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    def get(self, key: str) -> Optional[T]:
        """Returns None if missing or expired."""
        with self._lock:
            if key not in self._cache:
                self._misses += 1
                return None

            entry = self._cache[key]

            # Check TTL
            if self.ttl_seconds and (time.time() - entry.created_at) > self.ttl_seconds:
                self._remove_entry(key)
                self._misses += 1
                return None

            # Update access info and move to end
            entry.last_accessed = time.time()
            entry.access_count += 1
            self._cache.move_to_end(key)

            self._hits += 1
            return entry.value

    def set(
        self,
        key: str,
        value: T,
        size_bytes: Optional[int] = None,
        hash_value: Optional[str] = None,
    ) -> bool:
        """Store item, evicting LRU entries if needed. Returns False if item won't fit."""
        # Estimate size if not provided
        if size_bytes is None:
            if isinstance(value, (bytes, bytearray)):
                size_bytes = len(value)
            elif isinstance(value, str):
                size_bytes = len(value.encode('utf-8'))
            else:
                size_bytes = 1000  # Default estimate

        # Check if single item exceeds max size
        if size_bytes > self.max_size_bytes:
            return False

        with self._lock:
            # Remove existing entry if present
            if key in self._cache:
                self._remove_entry(key)

            # Evict until we have space
            while (
                (len(self._cache) >= self.max_items) or
                (self._total_size + size_bytes > self.max_size_bytes)
            ):
                if not self._cache:
                    break
                self._evict_one()

            # Add new entry
            now = time.time()
            entry = CacheEntry(
                value=value,
                size_bytes=size_bytes,
                created_at=now,
                last_accessed=now,
                access_count=0,
                hash=hash_value,
            )
            self._cache[key] = entry
            self._total_size += size_bytes

            return True

    def delete(self, key: str) -> bool:
        with self._lock:
            if key in self._cache:
                self._remove_entry(key)
                return True
            return False

    def clear(self):
        with self._lock:
            self._cache.clear()
            self._total_size = 0

    def _remove_entry(self, key: str):
        if key in self._cache:
            entry = self._cache.pop(key)
            self._total_size -= entry.size_bytes

    def _evict_one(self):
        if self._cache:
            # First item is LRU (OrderedDict maintains insertion order)
            key = next(iter(self._cache))
            self._remove_entry(key)
            self._evictions += 1

    def has_changed(self, key: str, new_hash: str) -> bool:
        with self._lock:
            if key not in self._cache:
                return True
            return self._cache[key].hash != new_hash

    @property
    def size(self) -> int:
        return len(self._cache)

    @property
    def size_bytes(self) -> int:
        return self._total_size

    @property
    def stats(self) -> Dict[str, Any]:
        total_requests = self._hits + self._misses
        hit_rate = self._hits / total_requests if total_requests > 0 else 0

        return {
            "items": len(self._cache),
            "size_bytes": self._total_size,
            "size_mb": round(self._total_size / (1024 * 1024), 2),
            "max_items": self.max_items,
            "max_size_mb": round(self.max_size_bytes / (1024 * 1024), 2),
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(hit_rate, 4),
            "evictions": self._evictions,
        }

    def cleanup_expired(self) -> int:
        if not self.ttl_seconds:
            return 0

        removed = 0
        now = time.time()

        with self._lock:
            expired_keys = [
                key for key, entry in self._cache.items()
                if (now - entry.created_at) > self.ttl_seconds
            ]

            for key in expired_keys:
                self._remove_entry(key)
                removed += 1

        return removed


class FeedCache:
    """Cache for feed images with status tracking and change detection."""

    def __init__(
        self,
        max_feeds: int = 5000,
        max_size_mb: int = 1024,  # 1 GB default
        ttl_seconds: Optional[float] = None,
    ):
        self._image_cache = LRUCache[bytes](
            max_items=max_feeds,
            max_size_bytes=max_size_mb * 1024 * 1024,
            ttl_seconds=ttl_seconds,
        )
        self._status: Dict[str, bool] = {}  # feed_id -> is_working
        self._vehicle_detected: Dict[str, bool] = {}  # feed_id -> has_vehicles
        self._image_hash: Dict[str, str] = {}  # feed_id -> hash
        # feed_id -> epoch seconds the frame was actually captured (not cached).
        # Detections are broadcast with this, so a slow cache cycle can never
        # attribute a detection to the wrong time.
        self._capture_time: Dict[str, float] = {}
        self._lock = Lock()

    def get_image(self, feed_id: str) -> Optional[bytes]:
        return self._image_cache.get(feed_id)

    def set_image(
        self,
        feed_id: str,
        image_bytes: bytes,
        is_working: bool = True,
        has_vehicles: bool = False,
        capture_time: Optional[float] = None,
    ):
        image_hash = hashlib.md5(image_bytes).hexdigest()

        self._image_cache.set(
            key=feed_id,
            value=image_bytes,
            size_bytes=len(image_bytes),
            hash_value=image_hash,
        )

        with self._lock:
            self._status[feed_id] = is_working
            self._vehicle_detected[feed_id] = has_vehicles
            self._image_hash[feed_id] = image_hash
            self._capture_time[feed_id] = capture_time if capture_time is not None else time.time()

    def get_capture_time(self, feed_id: str) -> Optional[float]:
        """Epoch seconds when this feed's cached frame was captured."""
        with self._lock:
            return self._capture_time.get(feed_id)

    def has_image_changed(self, feed_id: str, new_image_bytes: bytes) -> bool:
        new_hash = hashlib.md5(new_image_bytes).hexdigest()

        with self._lock:
            old_hash = self._image_hash.get(feed_id)

        return old_hash != new_hash

    def get_status(self, feed_id: str) -> Optional[bool]:
        with self._lock:
            return self._status.get(feed_id)

    def set_status(self, feed_id: str, is_working: bool):
        with self._lock:
            self._status[feed_id] = is_working

    def get_vehicle_detected(self, feed_id: str) -> bool:
        with self._lock:
            return self._vehicle_detected.get(feed_id, False)

    def set_vehicle_detected(self, feed_id: str, has_vehicles: bool):
        with self._lock:
            self._vehicle_detected[feed_id] = has_vehicles

    def get_all_status(self) -> Dict[str, bool]:
        with self._lock:
            return self._status.copy()

    def get_all_vehicle_detected(self) -> Dict[str, bool]:
        with self._lock:
            return self._vehicle_detected.copy()

    def delete(self, feed_id: str):
        self._image_cache.delete(feed_id)
        with self._lock:
            self._status.pop(feed_id, None)
            self._vehicle_detected.pop(feed_id, None)
            self._image_hash.pop(feed_id, None)

    def clear(self):
        self._image_cache.clear()
        with self._lock:
            self._status.clear()
            self._vehicle_detected.clear()
            self._image_hash.clear()

    @property
    def stats(self) -> Dict[str, Any]:
        base_stats = self._image_cache.stats

        with self._lock:
            working_count = sum(1 for v in self._status.values() if v)
            vehicle_count = sum(1 for v in self._vehicle_detected.values() if v)

        return {
            **base_stats,
            "feeds_working": working_count,
            "feeds_offline": len(self._status) - working_count,
            "feeds_with_vehicles": vehicle_count,
        }


async def periodic_cache_cleanup(cache: LRUCache, interval_seconds: float = 60.0):
    """Runs in background, removing expired entries."""
    while True:
        await asyncio.sleep(interval_seconds)
        try:
            removed = cache.cleanup_expired()
            if removed > 0:
                print(f"Cache cleanup: removed {removed} expired entries")
        except Exception as e:
            print(f"Cache cleanup error: {e}")
