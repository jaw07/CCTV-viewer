"""
Centralized app state - holds all shared data and components.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime
import asyncio
from threading import Lock

try:
    from .database import DatabaseManager
    from .tracker import TrackerManager
    from .websocket_manager import ConnectionManager
    from .core.cache import FeedCache
    from .observability import MetricsCollector, HealthChecker, StructuredLogger, CircuitBreaker, AlertManager
except ImportError:
    from database import DatabaseManager
    from tracker import TrackerManager
    from websocket_manager import ConnectionManager
    from core.cache import FeedCache
    from observability import MetricsCollector, HealthChecker, StructuredLogger, CircuitBreaker, AlertManager


@dataclass
class StreamConfig:
    enabled: bool = False
    format: str = "cot"
    ip: str = "127.0.0.1"
    port: int = 8087
    latticeToken: str = ""
    latticeSandboxToken: str = ""
    latticeIntegration: str = "taiwan-cctv"
    latticeUrl: str = ""
    # ChatSurfer fields
    chatsurferSession: str = ""
    chatsurferRoom: str = ""
    chatsurferNickname: str = "CCTV_Bot"
    chatsurferDomain: str = "chatsurferxmppunclass"
    chatsurferServerUrl: str = "http://localhost:8001"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "format": self.format,
            "ip": self.ip,
            "port": self.port,
            "latticeToken": self.latticeToken,
            "latticeSandboxToken": self.latticeSandboxToken,
            "latticeIntegration": self.latticeIntegration,
            "latticeUrl": self.latticeUrl,
            "chatsurferSession": self.chatsurferSession,
            "chatsurferRoom": self.chatsurferRoom,
            "chatsurferNickname": self.chatsurferNickname,
            "chatsurferDomain": self.chatsurferDomain,
            "chatsurferServerUrl": self.chatsurferServerUrl,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StreamConfig":
        return cls(
            enabled=data.get("enabled", False),
            format=data.get("format", "cot"),
            ip=data.get("ip", "127.0.0.1"),
            port=data.get("port", 8087),
            latticeToken=data.get("latticeToken", ""),
            latticeSandboxToken=data.get("latticeSandboxToken", ""),
            latticeIntegration=data.get("latticeIntegration", "taiwan-cctv"),
            latticeUrl=data.get("latticeUrl", ""),
            chatsurferSession=data.get("chatsurferSession", ""),
            chatsurferRoom=data.get("chatsurferRoom", ""),
            chatsurferNickname=data.get("chatsurferNickname", "CCTV_Bot"),
            chatsurferDomain=data.get("chatsurferDomain", "chatsurferxmppunclass"),
        )


class AppState:
    """
    Single source of truth for app-wide state.
    Thread-safe where needed.
    """

    def __init__(self):
        self._feeds_data: List[Dict] = []
        self._feeds_lock = Lock()

        self.feed_cache: Optional[FeedCache] = None

        self._feed_retry_after: Dict[str, float] = {}
        self._retry_lock = Lock()

        self._cycle_counter: int = 0
        self._last_update: float = 0

        # Core components - set during startup
        self.yolo_model = None
        self.db_manager: Optional[DatabaseManager] = None
        self.tracker_manager: Optional[TrackerManager] = None
        self.ws_manager: Optional[ConnectionManager] = None
        self.executor = None
        self.lattice_client = None

        self._stream_config = StreamConfig()
        self._stream_lock = Lock()

        # Observability
        self.metrics: Optional[MetricsCollector] = None
        self.logger: Optional[StructuredLogger] = None
        self.alert_manager: Optional[AlertManager] = None
        self.health_checker: Optional[HealthChecker] = None
        self.feed_circuit_breaker: Optional[CircuitBreaker] = None

        self.ssl_context = None
        self.device: str = "cpu"

        self._audit_log: List[Dict] = []
        self._audit_lock = Lock()

    # --- Feed Data ---

    @property
    def feeds_data(self) -> List[Dict]:
        with self._feeds_lock:
            return self._feeds_data.copy()

    @feeds_data.setter
    def feeds_data(self, value: List[Dict]):
        with self._feeds_lock:
            self._feeds_data = value

    def get_feed_by_id(self, feed_id: str) -> Optional[Dict]:
        with self._feeds_lock:
            return next((f for f in self._feeds_data if f['id'] == feed_id), None)

    @property
    def feeds_count(self) -> int:
        with self._feeds_lock:
            return len(self._feeds_data)

    # --- Feed Status (via FeedCache) ---

    def get_feed_status(self, feed_id: str) -> bool:
        if self.feed_cache:
            return self.feed_cache.get_status(feed_id) or False
        return False

    def set_feed_status(self, feed_id: str, is_working: bool):
        if self.feed_cache:
            self.feed_cache.set_status(feed_id, is_working)

    def get_all_feed_status(self) -> Dict[str, bool]:
        if self.feed_cache:
            return self.feed_cache.get_all_status()
        return {}

    def get_vehicle_detected(self, feed_id: str) -> bool:
        if self.feed_cache:
            return self.feed_cache.get_vehicle_detected(feed_id)
        return False

    def set_vehicle_detected(self, feed_id: str, has_vehicles: bool):
        if self.feed_cache:
            self.feed_cache.set_vehicle_detected(feed_id, has_vehicles)

    def get_all_vehicle_detected(self) -> Dict[str, bool]:
        if self.feed_cache:
            return self.feed_cache.get_all_vehicle_detected()
        return {}

    # --- Image Cache (via FeedCache) ---

    def get_cached_image(self, feed_id: str) -> Optional[bytes]:
        if self.feed_cache:
            return self.feed_cache.get_image(feed_id)
        return None

    def cache_image(self, feed_id: str, image_bytes: bytes,
                    is_working: bool = True, has_vehicles: bool = False,
                    capture_time: Optional[float] = None):
        if self.feed_cache:
            self.feed_cache.set_image(feed_id, image_bytes, is_working, has_vehicles,
                                      capture_time=capture_time)

    def get_capture_time(self, feed_id: str) -> Optional[float]:
        """When the cached frame for this feed was actually captured."""
        if self.feed_cache:
            return self.feed_cache.get_capture_time(feed_id)
        return None

    def has_image_changed(self, feed_id: str, new_image_bytes: bytes) -> bool:
        if self.feed_cache:
            return self.feed_cache.has_image_changed(feed_id, new_image_bytes)
        return True

    def get_cache_stats(self) -> Dict[str, Any]:
        if self.feed_cache:
            return self.feed_cache.stats
        return {}

    # --- Retry / Backoff ---

    def should_retry_feed(self, feed_id: str, current_time: float) -> bool:
        with self._retry_lock:
            retry_after = self._feed_retry_after.get(feed_id)
            if retry_after is None:
                return True
            return current_time >= retry_after

    def set_feed_backoff(self, feed_id: str, current_time: float, backoff_seconds: float):
        with self._retry_lock:
            self._feed_retry_after[feed_id] = current_time + backoff_seconds

    def clear_feed_backoff(self, feed_id: str):
        with self._retry_lock:
            self._feed_retry_after.pop(feed_id, None)
            self._feed_retry_after.pop(f"{feed_id}_interval", None)

    def get_backoff_interval(self, feed_id: str, default: float = 15.0, max_backoff: float = 300.0) -> float:
        with self._retry_lock:
            current = self._feed_retry_after.get(f"{feed_id}_interval", default)
            next_interval = min(current * 2, max_backoff)
            self._feed_retry_after[f"{feed_id}_interval"] = next_interval
            return current

    # --- Cycle Counter ---

    @property
    def cycle_counter(self) -> int:
        return self._cycle_counter

    def increment_cycle(self) -> int:
        self._cycle_counter += 1
        return self._cycle_counter

    @property
    def last_update(self) -> float:
        return self._last_update

    @last_update.setter
    def last_update(self, value: float):
        self._last_update = value

    # --- Stream Config ---

    @property
    def stream_config(self) -> StreamConfig:
        with self._stream_lock:
            return self._stream_config

    def update_stream_config(self, **kwargs) -> StreamConfig:
        with self._stream_lock:
            old_config = self._stream_config.to_dict()

            for key, value in kwargs.items():
                if hasattr(self._stream_config, key):
                    setattr(self._stream_config, key, value)

            new_config = self._stream_config.to_dict()

            self._log_audit("stream_config_update", {
                "old": old_config,
                "new": new_config,
                "changed_fields": [k for k in kwargs.keys() if old_config.get(k) != new_config.get(k)]
            })

            return self._stream_config

    # --- Audit Log ---

    def _log_audit(self, action: str, details: Dict[str, Any]):
        with self._audit_lock:
            entry = {
                "timestamp": datetime.utcnow().isoformat(),
                "action": action,
                "details": details
            }
            self._audit_log.append(entry)

            if len(self._audit_log) > 1000:
                self._audit_log = self._audit_log[-1000:]

            if self.logger:
                self.logger.info(f"AUDIT: {action}", **details)

    def get_audit_log(self, limit: int = 100) -> List[Dict]:
        with self._audit_lock:
            return self._audit_log[-limit:]

    # --- Stats ---

    def get_stats(self) -> Dict[str, Any]:
        feed_status = self.get_all_feed_status()
        vehicle_detected = self.get_all_vehicle_detected()
        cache_stats = self.get_cache_stats()

        working_count = sum(1 for s in feed_status.values() if s)
        vehicle_count = sum(1 for v in vehicle_detected.values() if v)

        stats = {
            "totalFeeds": self.feeds_count,
            "cachedFeeds": cache_stats.get("items", 0),
            "workingFeeds": working_count,
            "offlineFeeds": len(feed_status) - working_count,
            "vehiclesDetectedFeeds": vehicle_count,
            "lastUpdate": self.last_update,
            "cycleCounter": self.cycle_counter,
            "cacheSize": cache_stats.get("size_mb", 0),
            "cacheHitRate": cache_stats.get("hit_rate", 0),
            "device": self.device,
        }

        if self.ws_manager:
            stats["websocket"] = self.ws_manager.get_stats()

        return stats

    # --- Init Helpers ---

    def initialize_feed_cache(self, max_feeds: int = 5000, max_size_mb: int = 1024):
        self.feed_cache = FeedCache(max_feeds=max_feeds, max_size_mb=max_size_mb)

    def is_initialized(self) -> bool:
        return (
            self.yolo_model is not None and
            self.feed_cache is not None and
            self.feeds_count > 0
        )
