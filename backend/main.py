#!/usr/bin/env python3
"""
FastAPI backend for Taiwan CCTV Viewer
Ingests and caches all 2,500+ feeds in real-time

Features:
- Real-time vehicle detection with YOLOv8 (with GPU auto-detection)
- Multi-object tracking across camera feeds
- WebSocket real-time updates with optional authentication
- REST API with authentication and rate limiting
- Stream out to CoT/Lattice systems
- Centralized state management with AppState
- Comprehensive observability and audit logging
"""

import asyncio
import ssl
import certifi
import sys
import time
import socket
import hashlib
import json
import os
from datetime import datetime, timezone
from typing import Dict, List, Optional
from io import BytesIO
from pathlib import Path
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
import numpy as np

import httpx
import yaml
from fastapi import FastAPI, HTTPException, Request, Response, WebSocket, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import StreamingResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.exceptions import RequestValidationError
from PIL import Image
from ultralytics import YOLO
import cv2

# Import our modules
try:
    from .database import DatabaseManager
    from .tracker import TrackerManager, Detection as TrackerDetection
    from .websocket_manager import ConnectionManager, handle_websocket_messages
    from .observability import (
        MetricsCollector, HealthChecker, StructuredLogger,
        CircuitBreaker, AlertManager
    )
    from .middleware import (
        APIKeyMiddleware, RateLimitMiddleware, ErrorHandlerMiddleware,
        RequestLoggingMiddleware
    )
    from .middleware.error_handler import (
        custom_http_exception_handler, custom_validation_exception_handler
    )
    from .middleware import session_auth
    from . import youtube_frames
    from .schemas import (
        StreamConfigRequest, StreamConfigResponse,
        SearchResponse, FeedHistoryResponse, FeedStatsResponse,
        StatsResponse, HealthResponse, ErrorResponse
    )
    from .app_state import AppState, StreamConfig
    from .core.cache import FeedCache
    from .core.config import get_settings, Settings
except ImportError:
    # Fallback for direct execution
    from database import DatabaseManager
    from tracker import TrackerManager, Detection as TrackerDetection
    from websocket_manager import ConnectionManager, handle_websocket_messages
    from observability import (
        MetricsCollector, HealthChecker, StructuredLogger,
        CircuitBreaker, AlertManager
    )
    from middleware import (
        APIKeyMiddleware, RateLimitMiddleware, ErrorHandlerMiddleware,
        RequestLoggingMiddleware
    )
    from middleware.error_handler import (
        custom_http_exception_handler, custom_validation_exception_handler
    )
    from middleware import session_auth
    import youtube_frames
    from schemas import (
        StreamConfigRequest, StreamConfigResponse,
        SearchResponse, FeedHistoryResponse, FeedStatsResponse,
        StatsResponse, HealthResponse, ErrorResponse
    )
    from app_state import AppState, StreamConfig
    from core.cache import FeedCache
    from core.config import get_settings, Settings

# Import Prometheus client for metrics endpoint
from prometheus_client import make_asgi_app

# =============================================================================
# Configuration Loading with Pydantic Validation
# =============================================================================

def load_config() -> dict:
    """Load configuration from config.yaml"""
    config_path = Path(__file__).parent.parent / 'config.yaml'
    if not config_path.exists():
        print(f"Warning: config.yaml not found at {config_path}, using defaults")
        return {}

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    print(f"✓ Configuration loaded from {config_path}")
    return config


# Load and validate config
CONFIG = load_config()

# Get validated settings (Pydantic model)
try:
    settings = get_settings()
    print("✓ Configuration validated successfully")
except Exception as e:
    print(f"⚠ Configuration validation warning: {e}")
    settings = None

# =============================================================================
# Centralized Application State
# =============================================================================

# Single source of truth for all application state
app_state = AppState()

# Initialize observability components on app_state
app_state.metrics = MetricsCollector()
app_state.logger = StructuredLogger("cctv.main", log_level=CONFIG.get('logging', {}).get('level', 'INFO'))
app_state.alert_manager = AlertManager(CONFIG)
app_state.feed_circuit_breaker = CircuitBreaker(failure_threshold=10, recovery_timeout=300.0)

# Convenience references (for backward compatibility during transition)
logger = app_state.logger
metrics = app_state.metrics
alert_manager = app_state.alert_manager

# Detection settings from config
VEHICLE_CLASSES = CONFIG.get('detection', {}).get('vehicle_classes', [2, 5, 7])
# Coastal/harbour cameras see vessels, not cars. COCO class 8 is 'boat'.
if CONFIG.get('detection', {}).get('detect_boats', False) and 8 not in VEHICLE_CLASSES:
    VEHICLE_CLASSES = list(VEHICLE_CLASSES) + [8]

# Set up during startup when sources.cwa_coastal is enabled (see lifespan).
yt_grabber = None
_YT_CFG = CONFIG.get('sources', {}).get('cwa_coastal', {})
YT_ENABLED = _YT_CFG.get('enabled', False)
YT_WEST_ONLY = _YT_CFG.get('west_only', False)
MIN_BOX_SIZE = CONFIG.get('detection', {}).get('min_box_size', 20)
CONFIDENCE_THRESHOLD = CONFIG.get('detection', {}).get('confidence_threshold', 0.6)
SELECTIVE_SKIP_INTERVAL = CONFIG.get('performance', {}).get('selective_skip_interval', 2)

# =============================================================================
# SSL Configuration
# =============================================================================

def create_ssl_context(verify: bool = True) -> ssl.SSLContext:
    """
    Create SSL context with proper certificate verification.

    Args:
        verify: If True, use system CA certificates for verification.
                If False, disable verification (NOT recommended for production).

    Returns:
        Configured SSL context
    """
    if verify:
        # Use certifi's CA bundle for proper SSL verification
        ssl_context = ssl.create_default_context(cafile=certifi.where())
        ssl_context.check_hostname = True
        ssl_context.verify_mode = ssl.CERT_REQUIRED
    else:
        # Fallback for development/testing with self-signed certs
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        logger.warning("SSL verification disabled - NOT recommended for production")

    return ssl_context


# SSL context - defaults to verification enabled
# Set CCTV_SSL_VERIFY=false to disable (not recommended)
SSL_VERIFY = os.environ.get("CCTV_SSL_VERIFY", "true").lower() != "false"
ssl_context = create_ssl_context(verify=SSL_VERIFY)
app_state.ssl_context = ssl_context


# =============================================================================
# GPU Auto-Detection for YOLO
# =============================================================================

def detect_device() -> str:
    """
    Auto-detect the best available device for YOLO inference.

    Returns:
        'cuda' if GPU available, 'mps' for Apple Silicon, 'cpu' otherwise
    """
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            logger.info(f"GPU detected: {gpu_name}")
            return 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            logger.info("Apple Silicon GPU (MPS) detected")
            return 'mps'
        else:
            logger.info("No GPU detected, using CPU")
            return 'cpu'
    except ImportError:
        logger.warning("PyTorch not found, defaulting to CPU")
        return 'cpu'
    except Exception as e:
        logger.warning(f"Error detecting GPU: {e}, defaulting to CPU")
        return 'cpu'


def generate_cot_message(feed: Dict, capture_time: Optional[float] = None) -> str:
    """Generate a CoT XML message for a camera with vehicle detection.

    `capture_time` is when the analysed frame was actually captured. CoT
    consumers treat `time`/`start` as observation time, so using "now" would
    misdate the detection by however long the frame sat in the cache - which on
    a full feed cycle can be minutes. `now` is kept only for the stale window.
    """
    now = datetime.now(timezone.utc)
    observed = datetime.fromtimestamp(capture_time, timezone.utc) if capture_time else now
    time_str = observed.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'

    # Stale time: 1 hour after the message is produced.
    stale_time = datetime.fromtimestamp(now.timestamp() + 3600, timezone.utc)
    stale_str = stale_time.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'

    # Use feed ID as UID
    uid = feed['id']

    # Callsign: TrafficCam-{ID}
    callsign = f"TrafficCam-{feed['id']}"

    # Build remarks with location info
    remarks_parts = []
    if feed.get('roadName'):
        remarks_parts.append(f"Road: {feed['roadName']}")
    if feed.get('locationMile'):
        remarks_parts.append(f"Location: {feed['locationMile']}")
    if feed.get('description'):
        remarks_parts.append(feed['description'])
    remarks_parts.append("Vehicle Detected")
    if capture_time:
        remarks_parts.append(f"Observed: {time_str}")
    remarks = " | ".join(remarks_parts)

    # Get lat/lon
    lat = feed.get('lat', '0.0')
    lon = feed.get('lon', '0.0')

    # Build CoT XML
    cot_xml = f'''<?xml version='1.0' encoding='UTF-8' standalone='yes'?>
<event version='2.0' uid='{uid}' type='a-u-G' time='{time_str}' start='{time_str}' stale='{stale_str}' how='h-e'>
    <point lat='{lat}' lon='{lon}' hae='9999999.0' ce='9999999.0' le='9999999.0' />
    <detail>
        <contact callsign='{callsign}'/>
        <remarks>{remarks}</remarks>
        <link uid='{uid}' type='video' url='http://localhost:8001/api/feeds/{uid}/snapshot'/>
        <usericon iconsetpath='COT_MAPPING_2525B/a-u/a-u-G'/>
    </detail>
</event>'''

    return cot_xml


def send_cot_udp(message: str, ip: str, port: int):
    """Send CoT message via UDP"""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.sendto(message.encode('utf-8'), (ip, port))
        sock.close()
    except Exception as e:
        print(f"Error sending CoT message: {e}")


def publish_lattice_entity(feed: Dict, integration_name: str, annotated_image: bytes = None):
    """Publish entity to Lattice with vehicle detection and image"""
    global lattice_client

    if lattice_client is None:
        print("⚠ Lattice client is None, skipping publish")
        return

    try:
        from anduril import Location, Position, Aliases, Provenance, Ontology, MilView, Media, MediaItem, Enu
        from datetime import timedelta
        import math

        # Parse lat/lon
        lat = float(feed.get('lat', '0.0'))
        lon = float(feed.get('lon', '0.0'))

        # Create entity name/description
        name_parts = []
        if feed.get('roadName'):
            name_parts.append(feed['roadName'])
        if feed.get('locationMile'):
            name_parts.append(feed['locationMile'])
        entity_name = " - ".join(name_parts) if name_parts else feed['id']

        description = f"Traffic Camera: {entity_name} | Vehicle Detected"

        now = datetime.now(timezone.utc)
        expiry = now + timedelta(hours=1)

        entity_id = feed['id']
        entity_name_alias = f"TrafficCam-{entity_id}"

        print(f"Publishing entity {entity_id} to Lattice...")
        print(f"  Location: ({lat}, {lon})")
        print(f"  Description: {description}")
        print(f"  Alias: {entity_name_alias}")

        # Upload detection image to Objects API if provided
        media = None
        if annotated_image:
            try:
                # IMPORTANT: Use flat naming (no slashes) - slashes don't work in sandbox
                # Format: integration-entity_id-timestamp.jpg
                object_path = f"{integration_name}-{entity_id}-{now.strftime('%Y%m%d_%H%M%S')}.jpg"

                lattice_client.objects.upload_object(
                    object_path=object_path,
                    request=annotated_image
                )

                # Reference the uploaded object in Media
                # MediaItemType is a string literal union, use "MEDIA_TYPE_IMAGE" directly
                # relative_path should be the path relative to environment base URL
                media = Media(
                    media=[
                        MediaItem(
                            type="MEDIA_TYPE_IMAGE",
                            relative_path=f"api/v1/objects/{object_path}"
                        )
                    ]
                )
                print(f"  📷 Uploaded detection image to {object_path} ({len(annotated_image)} bytes)")
            except Exception as img_error:
                print(f"  ⚠ Failed to upload image: {img_error}")

        # Calculate unit velocity vector based on road direction for heading indication
        # We only know direction, not speed, so use unit vector (magnitude=1) for heading only
        direction = feed.get('direction', '').upper()
        velocity_enu = None
        if direction:
            # Map direction to unit velocity vector (magnitude = 1.0)
            # ENU: East-North-Up coordinate system
            # Using unit vector since we don't have actual speed data
            direction_map = {
                'E': Enu(e=1.0, n=0.0, u=0.0),      # East
                'W': Enu(e=-1.0, n=0.0, u=0.0),     # West
                'N': Enu(e=0.0, n=1.0, u=0.0),      # North
                'S': Enu(e=0.0, n=-1.0, u=0.0),     # South
                'NE': Enu(e=1.0/math.sqrt(2), n=1.0/math.sqrt(2), u=0.0),
                'NW': Enu(e=-1.0/math.sqrt(2), n=1.0/math.sqrt(2), u=0.0),
                'SE': Enu(e=1.0/math.sqrt(2), n=-1.0/math.sqrt(2), u=0.0),
                'SW': Enu(e=-1.0/math.sqrt(2), n=-1.0/math.sqrt(2), u=0.0),
            }
            velocity_enu = direction_map.get(direction)
            if velocity_enu:
                print(f"  🧭 Added heading: {direction}")

        # Publish entity using keyword arguments
        response = lattice_client.entities.publish_entity(
            entity_id=entity_id,
            description=description,
            is_live=True,
            created_time=now,
            expiry_time=expiry,
            location=Location(
                position=Position(
                    latitude_degrees=lat,
                    longitude_degrees=lon,
                    altitude_hae_meters=0.0
                ),
                velocity_enu=velocity_enu  # Unit vector for heading only, no speed data
            ),
            aliases=Aliases(
                name=entity_name_alias
            ),
            ontology=Ontology(
                template="TEMPLATE_TRACK",
                platform_type="VEHICLE"
            ),
            mil_view=MilView(
                disposition="DISPOSITION_UNKNOWN",
                environment="ENVIRONMENT_SURFACE"
            ),
            provenance=Provenance(
                integration_name=integration_name,
                data_type="CCTV",
                source_update_time=now
            ),
            media=media
        )

        print(f"✓ Successfully published entity {entity_id}")
    except Exception as e:
        print(f"✗ Error publishing to Lattice: {e}")
        import traceback
        print(f"  Traceback: {traceback.format_exc()}")


def initialize_lattice_client(env_token: str, sandbox_token: str = None, base_url: str = None):
    """
    Initialize Lattice client with tokens

    For Sandboxes:
    - env_token: Environment-level "Lattice Auth Token" (goes in Authorization: Bearer header)
    - sandbox_token: Account-level Sandboxes token (goes in anduril-sandbox-authorization header)

    For Production:
    - env_token: Your production Lattice token
    - sandbox_token: Not needed
    """
    global lattice_client
    try:
        from anduril import Lattice

        # Debug: Check if tokens are provided
        env_token_preview = f"{env_token[:10]}..." if len(env_token) > 10 else "EMPTY/SHORT"
        print(f"Initializing Lattice with environment token: {env_token_preview}")

        # Ensure base_url uses HTTPS
        if base_url and base_url.startswith("http://"):
            base_url = base_url.replace("http://", "https://")
            print(f"⚠ Fixed base_url protocol to HTTPS: {base_url}")

        # For sandbox environments, we need BOTH tokens
        # Use the SDK's headers parameter to pass the sandbox token
        custom_headers = {}
        if base_url and "sandbox" in base_url.lower():
            if sandbox_token:
                sandbox_token_preview = f"{sandbox_token[:10]}..." if len(sandbox_token) > 10 else "EMPTY/SHORT"
                print(f"Detected sandbox environment - using sandbox token: {sandbox_token_preview}")
                # IMPORTANT: Sandbox token also needs "Bearer " prefix!
                custom_headers["anduril-sandbox-authorization"] = f"Bearer {sandbox_token}"
            else:
                print("⚠ Warning: Sandbox URL detected but no sandbox token provided!")

        # Initialize with SDK's native headers parameter (better than custom httpx client)
        if custom_headers:
            if base_url:
                print(f"Using base_url: {base_url}")
                lattice_client = Lattice(
                    token=env_token,
                    base_url=base_url,
                    headers=custom_headers
                )
            else:
                lattice_client = Lattice(
                    token=env_token,
                    headers=custom_headers
                )
        else:
            # Standard initialization for production environments
            if base_url:
                print(f"Using base_url: {base_url}")
                lattice_client = Lattice(token=env_token, base_url=base_url)
            else:
                print("Using default Lattice URL")
                lattice_client = Lattice(token=env_token)

        print(f"✓ Lattice client initialized successfully")
    except Exception as e:
        print(f"✗ Error initializing Lattice client: {e}")
        lattice_client = None


def send_chatsurfer_message(feed: Dict, stream_cfg, annotated_image: bytes = None):
    """Send detection message to ChatSurfer with snapshot URL"""
    import requests

    url = "https://chatsurfer.nro.mil/api/chatserver/message"
    headers = {
        "cookie": f"SESSION={stream_cfg.chatsurferSession}",
        "Content-Type": "application/json"
    }

    # Build message content
    message_parts = ["[VEHICLE DETECTION]"]

    if feed.get('roadName'):
        message_parts.append(f"Road: {feed['roadName']}")
    if feed.get('locationMile'):
        message_parts.append(f"Location: {feed['locationMile']}")

    message_parts.append(f"Camera: {feed['id']}")

    if feed.get('lat') and feed.get('lon'):
        message_parts.append(f"Coords: {feed['lat']}, {feed['lon']}")

    # Add timestamp
    timestamp = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
    message_parts.append(f"Time: {timestamp}")

    # Add snapshot URL
    try:
        server_ip = socket.gethostbyname(socket.gethostname())
    except:
        server_ip = "localhost"
    feed_id = feed['id']
    message_parts.append(f"Snapshot: http://{server_ip}:8001/api/feeds/{feed_id}/snapshot")

    message = "\n".join(message_parts)

    payload = {
        "classification": "UNCLASSIFIED//FOUO",
        "message": message,
        "domainId": stream_cfg.chatsurferDomain,
        "nickName": stream_cfg.chatsurferNickname,
        "roomName": stream_cfg.chatsurferRoom
    }

    try:
        response = requests.post(url, headers=headers, json=payload, verify=False, timeout=10)
        if response.status_code == 204:
            print(f"✓ ChatSurfer message sent for feed {feed['id']}")
        else:
            print(f"✗ ChatSurfer error: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"✗ Error sending ChatSurfer message: {e}")


async def fetch_feed_list():
    """Fetch and parse the XML feed list from Taiwan Highway Bureau"""
    url = 'https://cctv-maintain.thb.gov.tw/opendataCCTVs.xml'

    # Use shared SSL context for proper verification
    async with httpx.AsyncClient(verify=ssl_context, timeout=30.0) as client:
        try:
            response = await client.get(url, headers={'User-Agent': 'Mozilla/5.0'})
            response.raise_for_status()

            # Parse XML
            root = ET.fromstring(response.text)

            # Extract namespace
            ns = {'ns': 'http://traffic.transportdata.tw/standard/traffic/schema/'}

            feeds = []
            for cctv in root.findall('.//ns:CCTV', ns):
                feed = {
                    'id': cctv.find('ns:CCTVID', ns).text if cctv.find('ns:CCTVID', ns) is not None else '',
                    'streamUrl': cctv.find('ns:VideoStreamURL', ns).text if cctv.find('ns:VideoStreamURL', ns) is not None else '',
                    'imageUrl': cctv.find('ns:VideoImageURL', ns).text if cctv.find('ns:VideoImageURL', ns) is not None else '',
                    'description': cctv.find('ns:SurveillanceDescription', ns).text if cctv.find('ns:SurveillanceDescription', ns) is not None else '',
                    'roadName': cctv.find('ns:RoadName', ns).text if cctv.find('ns:RoadName', ns) is not None else '',
                    'locationMile': cctv.find('ns:LocationMile', ns).text if cctv.find('ns:LocationMile', ns) is not None else '',
                    'lat': cctv.find('ns:PositionLat', ns).text if cctv.find('ns:PositionLat', ns) is not None else '',
                    'lon': cctv.find('ns:PositionLon', ns).text if cctv.find('ns:PositionLon', ns) is not None else '',
                    'direction': cctv.find('ns:RoadDirection', ns).text if cctv.find('ns:RoadDirection', ns) is not None else '',
                }
                if feed['id'] and feed['imageUrl']:
                    feeds.append(feed)

            # Update app_state instead of global variable
            if yt_grabber is not None:
                coastal = yt_grabber.build_feeds(west_only=YT_WEST_ONLY)
                # Coastal cameras go FIRST: there are only ~120 of them, their
                # frames are already in memory, and they are the feeds people
                # come here for. Ordering has to happen here rather than at
                # startup, because this runs again whenever the list refreshes -
                # appending them left them last in a 2400-feed cycle, so they
                # took many minutes to show any status.
                feeds = coastal + feeds
                if coastal:
                    logger.info("Added coastal camera feeds", count=len(coastal))

            app_state.feeds_data = feeds
            app_state.last_update = time.time()

            print(f"Loaded {len(feeds)} feeds from Taiwan Highway Bureau")
            return feeds

        except Exception as e:
            logger.error("Error fetching feed list", error=str(e))
            return []


def detect_vehicles(img_bytes: bytes, feed_id: str = None) -> tuple[bool, bytes, dict]:
    """Detect if vehicles are present in image using YOLO with tracking (optimized with OpenCV)
    Returns: (has_vehicles, annotated_image_bytes, detection_data)
    """
    try:
        if app_state.yolo_model is None:
            return False, img_bytes, {}

        # OPTIMIZATION: Use OpenCV for faster decoding (2-3x faster than PIL)
        nparr = np.frombuffer(img_bytes, np.uint8)
        img_array = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img_array is None:
            return False, img_bytes, {}

        # Run YOLO inference with confidence threshold
        results = app_state.yolo_model(img_array, verbose=False, conf=CONFIDENCE_THRESHOLD)[0]

        has_vehicles = False
        valid_boxes = []
        total_detections = 0
        filtered_detections = {"too_small": 0, "wrong_class": 0, "low_conf": 0}
        vehicle_types = []
        confidences = []

        # Check if any vehicle classes detected with proper filtering
        if results.boxes is not None and len(results.boxes) > 0:
            total_detections = len(results.boxes)

            # Filter to only vehicle detections with proper size and confidence
            for box in results.boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])

                # Only process vehicle classes
                if class_id not in VEHICLE_CLASSES:
                    filtered_detections["wrong_class"] += 1
                    continue

                # Check confidence threshold
                if confidence < CONFIDENCE_THRESHOLD:
                    filtered_detections["low_conf"] += 1
                    continue

                # Get box dimensions (x1, y1, x2, y2)
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                box_width = x2 - x1
                box_height = y2 - y1

                # Filter out tiny detections (likely false positives)
                if box_width < MIN_BOX_SIZE or box_height < MIN_BOX_SIZE:
                    filtered_detections["too_small"] += 1
                    continue

                # This is a valid vehicle detection
                has_vehicles = True
                valid_boxes.append(box)

                # Track vehicle type
                vehicle_type_map = {2: 'car', 5: 'bus', 7: 'truck', 3: 'motorcycle', 1: 'bicycle', 0: 'person'}
                vehicle_types.append(vehicle_type_map.get(class_id, f'class_{class_id}'))
                confidences.append(confidence)

            # DEBUG: Log detections
            if total_detections > 0 and feed_id:
                import random
                if random.random() < 0.01:  # Log 1% of detections to avoid spam
                    print(f"  [YOLO] {feed_id}: {total_detections} total, {len(valid_boxes)} vehicles, filtered: {filtered_detections}")

            # Update tracker if enabled
            tracked_vehicles = []
            track_counts = {}
            if has_vehicles and app_state.tracker_manager and feed_id:
                confirmed_tracks, track_counts = app_state.tracker_manager.update_tracker(feed_id, results)
                tracked_vehicles = [
                    {
                        "track_id": track.track_id,
                        "class": track.class_name,
                        "confidence": round(track.confidence, 2),
                        "bbox": track.bbox.tolist()
                    }
                    for track in confirmed_tracks
                ]

            # Prepare detection data
            detection_data = {
                "vehicle_count": len(valid_boxes),
                "vehicle_types": list(set(vehicle_types)),  # Unique types
                "confidence_avg": round(sum(confidences) / len(confidences), 2) if confidences else 0,
                "tracked_vehicles": tracked_vehicles,
                "track_counts": track_counts
            }

            if has_vehicles:
                # Get image with drawn boxes using YOLO's built-in plotting
                annotated = results.plot()

                # Draw track IDs if tracking enabled
                if tracked_vehicles:
                    for track in tracked_vehicles:
                        x1, y1, x2, y2 = track["bbox"]
                        cv2.putText(
                            annotated,
                            f"ID:{track['track_id']}",
                            (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 255, 0),
                            2
                        )

                # OPTIMIZATION: Use OpenCV for faster encoding
                _, encoded = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 75])
                return True, encoded.tobytes(), detection_data

        return False, img_bytes, {}
    except Exception as e:
        print(f"Error in vehicle detection: {e}")
        import traceback
        traceback.print_exc()
        return False, img_bytes, {}


async def fetch_snapshot(feed: Dict) -> Optional[bytes]:
    """Fetch a single snapshot from a feed"""
    url = feed['imageUrl']

    # YouTube-backed cameras are served from the frame grabber's cache rather
    # than over HTTP (see backend/youtube_frames.py).
    if url.startswith(youtube_frames.URL_SCHEME):
        if yt_grabber is None:
            return None
        return yt_grabber.get_frame(youtube_frames.YouTubeFrameGrabber.camera_id_from_url(url))

    # Use shared SSL context for proper verification
    async with httpx.AsyncClient(verify=ssl_context, timeout=10.0) as client:
        try:
            response = await client.get(url, headers={'User-Agent': 'Mozilla/5.0'})
            response.raise_for_status()

            # Verify it's a valid image
            img_data = response.content
            try:
                Image.open(BytesIO(img_data)).verify()
                return img_data
            except:
                return None

        except Exception as e:
            return None


async def update_feed_cache_worker():
    """Background worker - processes feeds in parallel batches (OPTIMIZED)"""
    logger.info("Starting optimized feed cache worker...")

    # Get performance settings from config
    config_batch_size = CONFIG.get('performance', {}).get('batch_size', 240)
    http_config = CONFIG.get('performance', {}).get('http', {})

    # Get current feeds from app_state
    feeds_data = app_state.feeds_data

    # OPTIMIZATION: Dynamic batch size based on feed count, capped by config
    BATCH_SIZE = min(config_batch_size, max(100, len(feeds_data) // 10))

    # OPTIMIZATION: Connection pool from config
    limits = httpx.Limits(
        max_keepalive_connections=http_config.get('max_keepalive', 250),
        max_connections=http_config.get('max_connections', 300),
        keepalive_expiry=http_config.get('keepalive_expiry', 60)
    )
    timeout = httpx.Timeout(
        http_config.get('timeout', 5.0),
        connect=http_config.get('connect_timeout', 2.0)
    )

    # Use shared SSL context for proper verification
    async with httpx.AsyncClient(verify=ssl_context, timeout=timeout, limits=limits) as client:
        while True:
            # Refresh feeds from app_state each cycle
            feeds_data = app_state.feeds_data
            if not feeds_data:
                await asyncio.sleep(5)
                continue

            cycle_num = app_state.increment_cycle()
            start_time = time.time()
            logger.info(f"Starting cycle #{cycle_num}", feeds=len(feeds_data), batch_size=BATCH_SIZE)

            # Fetch snapshot using shared client with exponential backoff
            async def fetch_with_client(feed):
                feed_id = feed['id']
                url = feed['imageUrl']
                current_time = time.time()

                # YouTube-backed cameras come from the grabber cache, not HTTP.
                if url.startswith(youtube_frames.URL_SCHEME):
                    if yt_grabber is None:
                        return feed_id, None, False, "grabber disabled"
                    cam = youtube_frames.YouTubeFrameGrabber.camera_id_from_url(url)
                    frame = yt_grabber.get_frame(cam)
                    return feed_id, frame, bool(frame), (None if frame else "no frame yet")

                # OPTIMIZATION: Skip feeds in backoff period (using app_state)
                if not app_state.should_retry_feed(feed_id, current_time):
                    return feed_id, None, False, "In backoff"

                try:
                    response = await client.get(url, headers={'User-Agent': 'Mozilla/5.0'})
                    response.raise_for_status()
                    img_data = response.content
                    if len(img_data) > 1000:
                        # Clear backoff on success
                        app_state.clear_feed_backoff(feed_id)
                        return feed_id, img_data, True, None
                    return feed_id, None, False, "Image too small"
                except httpx.TimeoutException as e:
                    return feed_id, None, False, f"Timeout: {str(e)}"
                except httpx.ConnectError as e:
                    return feed_id, None, False, f"Connection error: {str(e)}"
                except Exception as e:
                    return feed_id, None, False, f"Error: {type(e).__name__}"

            # Process feeds in sequential batches
            batches = [feeds_data[i:i+BATCH_SIZE] for i in range(0, len(feeds_data), BATCH_SIZE)]

            all_results = []
            error_counts = {}
            detection_stats = {"skipped_unchanged": 0, "skipped_selective": 0, "processed": 0}

            for batch_num, batch in enumerate(batches, 1):
                batch_start = time.time()
                tasks = [fetch_with_client(feed) for feed in batch]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                all_results.extend(results)

                # Get event loop for executor
                loop = asyncio.get_event_loop()

                # Update cache immediately after each batch completes
                for result in results:
                    if isinstance(result, tuple):
                        feed_id, img_data, success, error = result
                        if success and img_data:
                            # Use FeedCache's built-in change detection
                            image_changed = app_state.has_image_changed(feed_id, img_data)
                            should_detect = True

                            if not image_changed:
                                # Image unchanged, reuse previous detection
                                has_vehicles = app_state.get_vehicle_detected(feed_id)
                                annotated_img = app_state.get_cached_image(feed_id) or img_data
                                should_detect = False
                                detection_stats["skipped_unchanged"] += 1
                            elif cycle_num > 1 and not app_state.get_vehicle_detected(feed_id) and cycle_num % SELECTIVE_SKIP_INTERVAL != 0:
                                # OPTIMIZATION: Selective detection - only re-check "empty" feeds every 2 cycles
                                # BUT: Always process on first cycle to establish baseline
                                has_vehicles = False
                                annotated_img = img_data
                                should_detect = False
                                detection_stats["skipped_selective"] += 1

                            if should_detect:
                                # OPTIMIZATION: Run YOLO in thread pool to avoid blocking event loop
                                has_vehicles, annotated_img, detection_data = await loop.run_in_executor(
                                    app_state.executor, detect_vehicles, img_data, feed_id
                                )
                                detection_stats["processed"] += 1

                                # Save detection to database if enabled and vehicles detected
                                if has_vehicles and app_state.db_manager and detection_data:
                                    try:
                                        await app_state.db_manager.add_detection(
                                            feed_id=feed_id,
                                            vehicle_count=detection_data.get("vehicle_count", 0),
                                            vehicle_types=detection_data.get("vehicle_types", []),
                                            confidence_avg=detection_data.get("confidence_avg", 0)
                                        )
                                    except Exception as e:
                                        logger.error("Error saving detection to database", feed_id=feed_id, error=str(e))

                                # Send WebSocket update if vehicles detected
                                if has_vehicles and app_state.ws_manager and detection_data:
                                    try:
                                        await app_state.ws_manager.send_detection_update(feed_id, detection_data)
                                    except Exception as e:
                                        logger.error("Error sending WebSocket update", feed_id=feed_id, error=str(e))

                            # Store the annotated image (with boxes) in FeedCache
                            # For YouTube-backed cameras the frame was pulled by the
                            # grabber earlier; use that instant, not "now".
                            captured_at = time.time()
                            if yt_grabber is not None:
                                _fi = app_state.get_feed_by_id(feed_id)
                                _url = str(_fi.get('imageUrl', '')) if _fi else ''
                                if _url.startswith(youtube_frames.URL_SCHEME):
                                    cam = youtube_frames.YouTubeFrameGrabber.camera_id_from_url(_url)
                                    captured_at = yt_grabber.frame_time(cam) or captured_at
                            app_state.cache_image(feed_id, annotated_img, is_working=True,
                                                  has_vehicles=has_vehicles, capture_time=captured_at)

                            # Send streaming updates if enabled and vehicle detected
                            stream_cfg = app_state.stream_config
                            if has_vehicles and stream_cfg.enabled:
                                feed_info = app_state.get_feed_by_id(feed_id)
                                if feed_info:
                                    try:
                                        if stream_cfg.format == "cot":
                                            cot_msg = generate_cot_message(
                                                feed_info,
                                                capture_time=app_state.get_capture_time(feed_id),
                                            )
                                            send_cot_udp(cot_msg, stream_cfg.ip, stream_cfg.port)
                                        elif stream_cfg.format == "lattice":
                                            publish_lattice_entity(feed_info, stream_cfg.latticeIntegration, annotated_img)
                                        elif stream_cfg.format == "chatsurfer":
                                            send_chatsurfer_message(feed_info, stream_cfg, annotated_img)
                                    except Exception as e:
                                        logger.error(f"Error streaming {stream_cfg.format}", feed_id=feed_id, error=str(e))
                        else:
                            # Feed failed - update status
                            app_state.set_feed_status(feed_id, False)
                            app_state.set_vehicle_detected(feed_id, False)

                            # OPTIMIZATION: Exponential backoff for failed feeds
                            if error and "In backoff" not in error:
                                backoff = app_state.get_backoff_interval(feed_id)
                                app_state.set_feed_backoff(feed_id, time.time(), backoff)

                            if error:
                                error_type = error.split(':')[0]
                                error_counts[error_type] = error_counts.get(error_type, 0) + 1

                elapsed = time.time() - batch_start
                success = sum(1 for r in results if isinstance(r, tuple) and r[2])
                logger.debug(f"Batch {batch_num}/{len(batches)} done", elapsed=f"{elapsed:.2f}s", success=success, total=len(batch))

            elapsed = time.time() - start_time
            cache_stats = app_state.get_cache_stats()
            working_count = cache_stats.get("feeds_working", 0)
            vehicles_detected = cache_stats.get("feeds_with_vehicles", 0)

            logger.info(
                f"Cycle #{cycle_num} complete",
                elapsed=f"{elapsed:.2f}s",
                working=working_count,
                total=len(feeds_data),
                vehicles=vehicles_detected,
                processed=detection_stats['processed'],
                skipped_unchanged=detection_stats['skipped_unchanged'],
                skipped_selective=detection_stats['skipped_selective']
            )

            # Update metrics
            if metrics:
                metrics.feeds_total.set(len(feeds_data))
                metrics.feeds_online.set(working_count)
                metrics.feeds_with_vehicles.set(vehicles_detected)
                metrics.cache_size_bytes.set(cache_stats.get("size_bytes", 0))

            # Show error breakdown if any
            if error_counts:
                logger.warning("Feed errors in cycle", errors=error_counts)

            # Wait before next full update
            wait_time = max(0.5, 2.0 - elapsed)
            await asyncio.sleep(wait_time)


async def initialize_feeds():
    """Initialize feeds in background with prioritization"""
    global yt_grabber

    # Coastal YouTube cameras: resolve current video IDs before the first feed
    # fetch so they appear in the very first feed list.
    if YT_ENABLED:
        ok, why = youtube_frames.tools_available()
        if not ok:
            logger.warning("Coastal cameras disabled - required tools not found", detail=why)
        else:
            yt_grabber = youtube_frames.YouTubeFrameGrabber(
                frame_interval=_YT_CFG.get('frame_interval', 20),
                id_refresh_interval=_YT_CFG.get('id_refresh_interval', 3600),
                stream_latency=_YT_CFG.get('stream_latency_seconds', 0),
                max_concurrent=_YT_CFG.get('max_concurrent_grabs', 6),
                logger=logger,
            )
            pass  # resolution happens in the background, below

    # Fetch the road feed list FIRST and get the app serving. NOTHING involving
    # YouTube may block startup: resolving ids across ~37 channels and priming
    # ~120 frames takes minutes on a loaded host, and doing it first left the
    # whole app serving an empty feed list until it finished.
    await fetch_feed_list()

    # Resolve and prime coastal cameras in the background, then merge them into
    # the feed list. They appear a little after the road cameras rather than
    # holding up all ~2300 of them.
    if yt_grabber is not None:
        async def _bring_up_coastal():
            loop = asyncio.get_event_loop()
            try:
                await loop.run_in_executor(None, yt_grabber.refresh_ids)
                # Merge the coastal feeds in now that they are known.
                await fetch_feed_list()
                cams = list(yt_grabber.resolved)
                step = yt_grabber.max_concurrent
                for i in range(0, len(cams), step):
                    await asyncio.gather(*(
                        loop.run_in_executor(None, yt_grabber.grab, c)
                        for c in cams[i:i + step]
                    ), return_exceptions=True)
                logger.info("coastal cameras ready", cameras=len(cams))
                asyncio.create_task(yt_grabber.run())
            except Exception as exc:
                logger.error("coastal bring-up failed", error=str(exc))
        asyncio.create_task(_bring_up_coastal())

    feeds_data = app_state.feeds_data

    # OPTIMIZATION: Prioritize major highways (國道) for faster initial cache warm-up
    if feeds_data:
        # Coastal cameras first (few, and the frames are already in memory),
        # then major highways (國道), then everything else.
        coastal = [f for f in feeds_data if f.get('id', '').startswith('CWA-')]
        rest = [f for f in feeds_data if not f.get('id', '').startswith('CWA-')]
        priority_feeds = [f for f in rest if '國道' in f.get('roadName', '')]
        other_feeds = [f for f in rest if '國道' not in f.get('roadName', '')]
        # Same ordering as fetch_feed_list applies: coastal, then major highways.
        app_state.feeds_data = coastal + priority_feeds + other_feeds
        logger.info("Feed prioritization complete", coastal=len(coastal),
                    priority=len(priority_feeds), other=len(other_feeds))

    # Sync feeds to database if enabled
    if app_state.db_manager and feeds_data:
        logger.info(f"Syncing {len(feeds_data)} feeds to database...")
        synced = 0
        for feed in feeds_data[:100]:  # Sync first 100 to avoid blocking
            try:
                await app_state.db_manager.upsert_feed(feed)
                synced += 1
            except Exception as e:
                logger.error("Error syncing feed", feed_id=feed.get('id'), error=str(e))
        logger.info("Feed sync complete", synced=synced)

    # Start background cache worker
    asyncio.create_task(update_feed_cache_worker())

    # Refresh feed list every hour
    async def refresh_feed_list():
        while True:
            await asyncio.sleep(3600)  # 1 hour
            await fetch_feed_list()

            # Sync new feeds to database
            feeds_data = app_state.feeds_data
            if app_state.db_manager and feeds_data:
                for feed in feeds_data[:50]:  # Sync subset on refresh
                    try:
                        await app_state.db_manager.upsert_feed(feed)
                    except Exception:
                        pass

    asyncio.create_task(refresh_feed_list())


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events"""
    import os

    # Attach app_state to FastAPI app for access in endpoints
    app.state.cctv = app_state

    # Initialize health checker with app_state
    app_state.health_checker = HealthChecker(app_state)

    # Startup
    logger.info("Starting Taiwan CCTV API (OPTIMIZED)...")

    # ==========================================================================
    # GPU Auto-Detection for YOLO
    # ==========================================================================
    device = detect_device()
    app_state.device = device
    logger.info(f"YOLO inference device: {device}")

    # Load YOLO model with appropriate device
    logger.info("Loading YOLOv8n model...")
    try:
        yolo_model = YOLO('yolov8n.pt')  # Nano model for speed

        # Move model to detected device (GPU if available)
        if device in ('cuda', 'mps'):
            yolo_model.to(device)
            logger.info(f"YOLO model loaded on {device.upper()}")
        else:
            logger.info("YOLO model loaded on CPU")

        app_state.yolo_model = yolo_model

        # Set system info metrics
        metrics.system_info.info({
            'version': '1.0.0',
            'model': 'yolov8n',
            'device': device,
            'python_version': str(sys.version_info[:3])
        })
    except Exception as e:
        logger.error("Could not load YOLO model", error=str(e))
        logger.warning("Vehicle detection will be disabled")
        await alert_manager.send_alert(
            alert_type="startup_failure",
            severity="critical",
            message="Failed to load YOLO model",
            metadata={"error": str(e)}
        )

    # ==========================================================================
    # Initialize FeedCache (using LRUCache)
    # ==========================================================================
    cache_config = CONFIG.get('cache', {})
    app_state.initialize_feed_cache(
        max_feeds=cache_config.get('max_feeds', 5000),
        max_size_mb=cache_config.get('max_size_mb', 1024)
    )
    logger.info("FeedCache initialized", max_size_mb=cache_config.get('max_size_mb', 1024))

    # ==========================================================================
    # Initialize ThreadPoolExecutor
    # ==========================================================================
    max_workers = CONFIG.get('performance', {}).get('worker_threads', min(os.cpu_count() or 4, 8))
    app_state.executor = ThreadPoolExecutor(max_workers=max_workers)
    logger.info("ThreadPoolExecutor initialized", max_workers=max_workers)

    # ==========================================================================
    # Initialize Database
    # ==========================================================================
    db_config = CONFIG.get('database', {})
    if db_config.get('enabled', True):
        db_type = db_config.get('type', 'sqlite')
        if db_type == 'sqlite':
            db_path = db_config.get('path', './data/cctv_data.db')
            # Ensure data directory exists
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
            database_url = f"sqlite+aiosqlite:///{db_path}"
        else:
            # PostgreSQL/MySQL configuration
            host = db_config.get('host', 'localhost')
            port = db_config.get('port', 5432)
            username = db_config.get('username', 'user')
            password = db_config.get('password', 'pass')
            database = db_config.get('database', 'cctv')
            database_url = f"postgresql+asyncpg://{username}:{password}@{host}:{port}/{database}"

        try:
            db_manager = DatabaseManager(database_url)
            await db_manager.init_db()
            app_state.db_manager = db_manager
            logger.info("Database initialized", db_type=db_type)

            # Schedule periodic cleanup with logging and metrics
            async def cleanup_task():
                while True:
                    await asyncio.sleep(86400)  # Once per day
                    retention = db_config.get('retention', {})
                    try:
                        logger.info("Starting scheduled database cleanup...")
                        deleted = await app_state.db_manager.cleanup_old_data(
                            detection_retention_days=retention.get('detections', 30),
                            history_retention_days=retention.get('feed_history', 90)
                        )
                        logger.info("Database cleanup completed", records_deleted=deleted)

                        # Log audit entry for cleanup
                        app_state._log_audit("database_cleanup", {
                            "retention_days": retention.get('detections', 30),
                            "records_deleted": deleted
                        })
                    except Exception as e:
                        logger.error("Database cleanup failed", error=str(e))

            asyncio.create_task(cleanup_task())
        except Exception as e:
            logger.error("Could not initialize database", error=str(e))
            await alert_manager.send_alert(
                alert_type="database_init_failure",
                severity="error",
                message="Failed to initialize database",
                metadata={"error": str(e)}
            )
    else:
        logger.info("Database disabled in config")

    # ==========================================================================
    # Initialize Tracker
    # ==========================================================================
    tracking_config = CONFIG.get('detection', {}).get('tracking', {})
    if tracking_config.get('enabled', True):
        app_state.tracker_manager = TrackerManager(
            max_age=tracking_config.get('max_age', 30),
            min_hits=tracking_config.get('min_hits', 3),
            iou_threshold=tracking_config.get('iou_threshold', 0.3)
        )
        logger.info("Vehicle tracking initialized")
    else:
        logger.info("Vehicle tracking disabled in config")

    # ==========================================================================
    # Initialize WebSocket Manager
    # ==========================================================================
    app_state.ws_manager = ConnectionManager()
    logger.info("WebSocket manager initialized")

    # Start background tasks
    asyncio.create_task(initialize_feeds())

    # WebSocket heartbeat task
    websocket_config = CONFIG.get('websocket', {})
    if websocket_config.get('enabled', True):
        async def heartbeat_task():
            interval = websocket_config.get('heartbeat_interval', 30)
            while True:
                await asyncio.sleep(interval)
                await app_state.ws_manager.send_heartbeat()

        asyncio.create_task(heartbeat_task())
        logger.info("WebSocket heartbeat started", interval=websocket_config.get('heartbeat_interval', 30))

    # Metrics update task (using app_state)
    async def metrics_update_task():
        """Periodically update Prometheus metrics"""
        while True:
            await asyncio.sleep(10)  # Update every 10 seconds
            try:
                stats = app_state.get_stats()
                metrics.feeds_total.set(stats.get("totalFeeds", 0))
                metrics.feeds_online.set(stats.get("workingFeeds", 0))
                metrics.feeds_with_vehicles.set(stats.get("vehiclesDetectedFeeds", 0))
                metrics.cache_size_bytes.set(stats.get("cacheSize", 0) * 1024 * 1024)
                if app_state.ws_manager:
                    metrics.active_websockets.set(len(app_state.ws_manager.active_connections))
            except Exception as e:
                logger.error("Error updating metrics", error=str(e))

    asyncio.create_task(metrics_update_task())
    logger.info("Metrics update task started")

    yield

    # ==========================================================================
    # Shutdown
    # ==========================================================================
    logger.info("Shutting down Taiwan CCTV API...")
    if app_state.executor:
        app_state.executor.shutdown(wait=True)
        logger.info("ThreadPoolExecutor shut down")
    if app_state.db_manager:
        await db_manager.close()
        logger.info("Database connection closed")


app = FastAPI(
    title="Taiwan CCTV API",
    description="Real-time vehicle detection and tracking for Taiwan highway cameras",
    version="2.0.0",
    lifespan=lifespan,
    responses={
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Invalid API key"},
        422: {"model": ErrorResponse, "description": "Validation error"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)

# Register custom exception handlers
app.add_exception_handler(HTTPException, custom_http_exception_handler)
app.add_exception_handler(RequestValidationError, custom_validation_exception_handler)

# ============================================================================
# Middleware Stack (order matters - first added = outermost)
# ============================================================================

# 1. Error handling (outermost - catches all errors)
app.add_middleware(
    ErrorHandlerMiddleware,
    include_traceback=os.environ.get("CCTV_DEBUG", "false").lower() == "true"
)

# 2. Request logging
app.add_middleware(
    RequestLoggingMiddleware,
    exclude_paths=["/health", "/health/live", "/health/ready", "/metrics"],
    slow_request_threshold_ms=1000.0,
)

# 3. Rate limiting
rate_limit_config = CONFIG.get('api', {})
app.add_middleware(
    RateLimitMiddleware,
    requests_per_minute=rate_limit_config.get('rate_limit', 100),
    burst_size=rate_limit_config.get('burst_size', 20),
    enabled=rate_limit_config.get('rate_limit_enabled', True),
)

# 4. API key authentication
app.add_middleware(
    APIKeyMiddleware,
    exclude_paths=["/", "/health", "/health/live", "/health/ready", "/docs", "/redoc", "/openapi.json"],
    exclude_prefixes=["/metrics"],
    enabled=os.environ.get("CCTV_AUTH_ENABLED", "true").lower() == "true",
)

# 4b. Browser session authentication (login form + signed cookie)
app.add_middleware(
    session_auth.SessionAuthMiddleware,
    exclude_paths=[
        "/", "/login.html", "/health", "/health/live", "/health/ready",
        "/docs", "/redoc", "/openapi.json",
        "/api/auth/login", "/api/auth/logout",
    ],
    exclude_prefixes=["/metrics", "/styles/", "/scripts/"],
    enabled=os.environ.get("CCTV_AUTH_ENABLED", "true").lower() == "true",
)

# 4c. Prevent caching of HTML/API/redirects so auth state is never served stale
app.add_middleware(session_auth.NoStoreMiddleware)

# 5. GZip compression
app.add_middleware(GZipMiddleware, minimum_size=1000)

# 6. CORS - Configure properly based on environment
cors_config = CONFIG.get('api', {})
allowed_origins = os.environ.get("CCTV_CORS_ORIGINS", "").split(",")
allowed_origins = [o.strip() for o in allowed_origins if o.strip()]

# Default to restrictive CORS in production, permissive in development
if not allowed_origins:
    if os.environ.get("CCTV_ENV", "development") == "production":
        allowed_origins = ["https://your-frontend-domain.com"]  # Configure this!
    else:
        allowed_origins = ["*"]  # Development only

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True if allowed_origins != ["*"] else False,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["X-Request-ID", "X-RateLimit-Limit", "X-RateLimit-Remaining", "X-RateLimit-Reset"],
)

# Mount Prometheus metrics endpoint
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)


@app.get("/")
async def root():
    """Entry point: send browsers to the UI (which gates on login)."""
    return RedirectResponse("/index.html", status_code=302)


@app.get("/api")
async def api_root():
    """API root"""
    stats = app_state.get_stats()
    return {
        "name": "Taiwan CCTV API",
        "version": "2.0.0",
        "feeds": stats.get("totalFeeds", 0),
        "cached": stats.get("cachedFeeds", 0),
        "working": stats.get("workingFeeds", 0),
        "device": app_state.device,
    }


@app.post("/api/auth/login")
async def auth_login(request: Request):
    """Exchange username/password for a signed session cookie."""
    client_ip = request.headers.get("X-Forwarded-For", "").split(",")[0].strip() \
        or (request.client.host if request.client else "unknown")

    if session_auth.is_login_throttled(client_ip):
        raise HTTPException(
            status_code=429,
            detail="Too many failed sign-in attempts. Try again in a few minutes.",
        )

    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid request body")

    username = (body or {}).get("username", "")
    password = (body or {}).get("password", "")
    if not isinstance(username, str) or not isinstance(password, str):
        raise HTTPException(status_code=400, detail="Invalid credentials")

    if not session_auth.authenticate(username, password):
        session_auth.record_failed_login(client_ip)
        logger.warning("failed_login", extra={"username": username, "client_ip": client_ip})
        raise HTTPException(status_code=401, detail="Invalid username or password")

    session_auth.clear_failed_logins(client_ip)
    logger.info("login_success", extra={"username": username, "client_ip": client_ip})

    response = JSONResponse({"status": "ok", "username": username})
    response.set_cookie(
        session_auth.COOKIE_NAME,
        session_auth.create_session_token(username),
        max_age=session_auth.SESSION_TTL,
        **session_auth.cookie_kwargs(),
    )
    return response


@app.post("/api/auth/logout")
async def auth_logout():
    """Clear the session cookie."""
    response = JSONResponse({"status": "ok"})
    response.delete_cookie(session_auth.COOKIE_NAME, path="/")
    return response


@app.get("/api/auth/me")
async def auth_me(request: Request):
    """Return the signed-in user, or auth_required=False when auth is off."""
    if not session_auth.get_users() or \
            os.environ.get("CCTV_AUTH_ENABLED", "true").lower() != "true":
        return {"auth_required": False, "username": None}
    username = session_auth.verify_session_token(
        request.cookies.get(session_auth.COOKIE_NAME)
    )
    if not username:
        raise HTTPException(status_code=401, detail="Login required")
    return {"auth_required": True, "username": username}


# The feed catalogue is static once loaded, but it is ~90% of a 700KB payload.
# Re-serialising it per request burned CPU that the detection threads also need,
# so the static half is rendered once and reused; only the live status is rebuilt.
_feeds_blob_cache: Dict[str, object] = {"key": None, "blob": b""}


@app.get("/api/feeds")
async def get_feeds():
    """Get all feed metadata"""
    cache_key = (id(app_state.feeds_data), len(app_state.feeds_data))
    if _feeds_blob_cache["key"] != cache_key:
        _feeds_blob_cache["blob"] = json.dumps(
            app_state.feeds_data, separators=(",", ":")
        ).encode()
        _feeds_blob_cache["key"] = cache_key

    def dump(value) -> bytes:
        return json.dumps(value, separators=(",", ":")).encode()

    body = b"".join([
        b'{"feeds":', _feeds_blob_cache["blob"],
        b',"status":', dump(app_state.get_all_feed_status()),
        b',"vehicleDetected":', dump(app_state.get_all_vehicle_detected()),
        b',"lastUpdate":', dump(app_state.last_update),
        b"}",
    ])
    return Response(content=body, media_type="application/json")


@app.get("/api/feeds/status")
async def get_feeds_status():
    """Live status only.

    The UI polls this every couple of seconds. It used to poll /api/feeds, which
    ships the whole 700KB catalogue each time purely to read these two maps.
    """
    return {
        "status": app_state.get_all_feed_status(),
        "vehicleDetected": app_state.get_all_vehicle_detected(),
        "lastUpdate": app_state.last_update,
    }


@app.get("/api/feeds/{feed_id}/snapshot")
async def get_snapshot(feed_id: str):
    """Get latest cached snapshot for a feed"""
    cached_image = app_state.get_cached_image(feed_id)
    if cached_image is None:
        raise HTTPException(status_code=404, detail="Feed not found or offline")

    headers = {}
    captured_at = app_state.get_capture_time(feed_id)
    if captured_at:
        observed = datetime.fromtimestamp(captured_at, timezone.utc)
        # ISO for machines, Last-Modified for HTTP caches, age for quick checks.
        headers["X-Capture-Time"] = observed.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'
        headers["X-Capture-Age-Seconds"] = str(int(time.time() - captured_at))
        # 'image' = read from the camera's own on-frame clock (authoritative);
        # 'download' = when we pulled it (fallback, may lag real time).
        src = None
        if yt_grabber is not None:
            _f = app_state.get_feed_by_id(feed_id)
            _u = str(_f.get('imageUrl', '')) if _f else ''
            if _u.startswith(youtube_frames.URL_SCHEME):
                src = yt_grabber.frame_time_source(
                    youtube_frames.YouTubeFrameGrabber.camera_id_from_url(_u))
        headers["X-Capture-Time-Source"] = src or "download"
        headers["Last-Modified"] = observed.strftime('%a, %d %b %Y %H:%M:%S GMT')

    return Response(content=cached_image, media_type="image/jpeg", headers=headers)


@app.get("/api/feeds/{feed_id}/stream")
async def get_stream(feed_id: str):
    """Get MJPEG stream for a feed"""
    # Find feed
    feed = app_state.get_feed_by_id(feed_id)
    if not feed:
        raise HTTPException(status_code=404, detail="Feed not found")

    url = feed['streamUrl']

    async def stream_generator():
        # Use shared SSL context for proper verification
        async with httpx.AsyncClient(verify=ssl_context, timeout=30.0) as client:
            try:
                async with client.stream('GET', url, headers={'User-Agent': 'Mozilla/5.0'}) as response:
                    async for chunk in response.aiter_bytes(chunk_size=4096):
                        yield chunk
            except Exception as e:
                logger.error("Stream error", feed_id=feed_id, error=str(e))

    return StreamingResponse(
        stream_generator(),
        media_type="multipart/x-mixed-replace; boundary=--DIGIEVER"
    )


@app.get("/health")
async def health_check():
    """
    Comprehensive health check endpoint for monitoring and alerting.
    Returns detailed status of all system components.
    """
    if app_state.health_checker is None:
        return {
            "status": "unhealthy",
            "message": "Health checker not initialized",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

    try:
        health_status = await app_state.health_checker.check_all()
        return health_status
    except Exception as e:
        logger.error("Health check failed", error=str(e))
        return {
            "status": "unhealthy",
            "message": f"Health check error: {str(e)}",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }


@app.get("/health/live")
async def liveness_probe():
    """
    Liveness probe for Kubernetes/container orchestration.
    Returns 200 if the application is running.
    """
    return {"status": "alive", "timestamp": datetime.now(timezone.utc).isoformat()}


@app.get("/health/ready")
async def readiness_probe():
    """
    Readiness probe for Kubernetes/container orchestration.
    Returns 200 if the application is ready to serve traffic.
    """
    # Check critical components using app_state
    is_ready = app_state.is_initialized()

    if is_ready:
        return {
            "status": "ready",
            "feeds_loaded": app_state.feeds_count,
            "device": app_state.device,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    else:
        raise HTTPException(
            status_code=503,
            detail={
                "status": "not_ready",
                "yolo_loaded": app_state.yolo_model is not None,
                "feeds_loaded": app_state.feeds_count,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        )


@app.get("/api/stats")
async def get_stats():
    """Get system statistics"""
    return app_state.get_stats()


@app.get("/api/operational/status")
async def get_operational_status():
    """
    Operational status dashboard endpoint.
    Provides comprehensive system metrics for monitoring and troubleshooting.
    """
    stats = app_state.get_stats()
    cache_stats = app_state.get_cache_stats()

    # Calculate uptime metrics
    import psutil
    process = psutil.Process()
    uptime_seconds = time.time() - process.create_time()

    operational_status = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "uptime_seconds": int(uptime_seconds),
        "uptime_human": f"{int(uptime_seconds // 3600)}h {int((uptime_seconds % 3600) // 60)}m",

        "feeds": {
            "total": stats.get("totalFeeds", 0),
            "online": stats.get("workingFeeds", 0),
            "offline": stats.get("offlineFeeds", 0),
            "online_percentage": round((stats.get("workingFeeds", 0) / stats.get("totalFeeds", 1) * 100), 2),
            "with_vehicles": stats.get("vehiclesDetectedFeeds", 0),
            "last_update": stats.get("lastUpdate", 0)
        },

        "cache": {
            "size_mb": cache_stats.get("size_mb", 0),
            "items": cache_stats.get("items", 0),
            "hit_rate": cache_stats.get("hit_rate", 0),
            "evictions": cache_stats.get("evictions", 0)
        },

        "components": {
            "yolo_model": "loaded" if app_state.yolo_model else "not_loaded",
            "device": app_state.device,
            "database": "enabled" if app_state.db_manager else "disabled",
            "tracker": "enabled" if app_state.tracker_manager else "disabled",
            "websocket": "enabled" if app_state.ws_manager else "disabled"
        },

        "websocket": {
            "active_connections": len(app_state.ws_manager.active_connections) if app_state.ws_manager else 0,
            "stats": app_state.ws_manager.get_stats() if app_state.ws_manager else {}
        },

        "circuit_breakers": {
            "feed_fetcher": app_state.feed_circuit_breaker.get_state() if app_state.feed_circuit_breaker else {}
        },

        "alerts": {
            "recent": alert_manager.get_recent_alerts(limit=10)
        },

        "audit_log": {
            "recent": app_state.get_audit_log(limit=5)
        }
    }

    # Add database stats if available
    if app_state.db_manager:
        try:
            from database import Detection as DetectionModel, VehicleTrack as VehicleTrackModel
            from sqlalchemy import func, select
            async with app_state.db_manager.session() as session:
                detection_count = await session.execute(
                    select(func.count()).select_from(DetectionModel)
                )
                track_count = await session.execute(
                    select(func.count()).select_from(VehicleTrackModel)
                )

                operational_status["database"] = {
                    "detections_count": detection_count.scalar(),
                    "tracks_count": track_count.scalar()
                }
        except Exception as e:
            logger.error("Failed to get database stats", error=str(e))
            operational_status["database"] = {"error": str(e)}

    return operational_status


@app.get("/api/feeds/{feed_id}/history")
async def get_feed_history(feed_id: str, hours: int = 24):
    """Get detection history for a specific feed"""
    if not app_state.db_manager:
        raise HTTPException(status_code=503, detail="Database not enabled")

    try:
        detections = await app_state.db_manager.get_feed_history(feed_id, hours)
        return {
            "feed_id": feed_id,
            "hours": hours,
            "detections": [
                {
                    "id": d.id,
                    "timestamp": d.timestamp.isoformat(),
                    "vehicle_count": d.vehicle_count,
                    "vehicle_types": d.vehicle_types,
                    "confidence_avg": d.confidence_avg
                }
                for d in detections
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching history: {str(e)}")


@app.get("/api/feeds/{feed_id}/stats")
async def get_feed_stats(feed_id: str, days: int = 7):
    """Get statistics for a specific feed"""
    if not app_state.db_manager:
        raise HTTPException(status_code=503, detail="Database not enabled")

    try:
        stats = await app_state.db_manager.get_feed_stats(feed_id, days)
        return {
            "feed_id": feed_id,
            **stats
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching stats: {str(e)}")


@app.get("/api/search")
async def search_feeds(
    road: Optional[str] = None,
    has_vehicles: Optional[bool] = None,
    lat_min: Optional[float] = None,
    lat_max: Optional[float] = None,
    lon_min: Optional[float] = None,
    lon_max: Optional[float] = None
):
    """Search feeds with filters"""
    results = app_state.feeds_data.copy()
    vehicle_detected = app_state.get_all_vehicle_detected()
    feed_status = app_state.get_all_feed_status()

    # Filter by road name
    if road:
        results = [f for f in results if road.lower() in f.get('roadName', '').lower()]

    # Filter by vehicle detection
    if has_vehicles is not None:
        results = [
            f for f in results
            if vehicle_detected.get(f['id'], False) == has_vehicles
        ]

    # Filter by bounding box
    if lat_min is not None and lat_max is not None and lon_min is not None and lon_max is not None:
        results = [
            f for f in results
            if (lat_min <= float(f.get('lat', 0) or 0) <= lat_max and
                lon_min <= float(f.get('lon', 0) or 0) <= lon_max)
        ]

    # Add status and vehicle detection to results
    enriched_results = []
    for feed in results:
        feed_copy = feed.copy()
        feed_copy['status'] = feed_status.get(feed['id'], False)
        feed_copy['vehicleDetected'] = vehicle_detected.get(feed['id'], False)
        enriched_results.append(feed_copy)

    return {
        "query": {
            "road": road,
            "has_vehicles": has_vehicles,
            "bbox": [lat_min, lon_min, lat_max, lon_max] if all([lat_min, lat_max, lon_min, lon_max]) else None
        },
        "count": len(enriched_results),
        "feeds": enriched_results
    }


@app.get("/api/map")
async def get_map_data():
    """Get feed data formatted for map display"""
    map_features = []
    feed_status = app_state.get_all_feed_status()
    vehicle_detected = app_state.get_all_vehicle_detected()

    for feed in app_state.feeds_data:
        lat = float(feed.get('lat', 0) or 0)
        lon = float(feed.get('lon', 0) or 0)

        if lat == 0 or lon == 0:
            continue

        feature = {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [lon, lat]
            },
            "properties": {
                "id": feed['id'],
                "roadName": feed.get('roadName', ''),
                "locationMile": feed.get('locationMile', ''),
                "description": feed.get('description', ''),
                "direction": feed.get('direction', ''),
                "isWorking": feed_status.get(feed['id'], False),
                "hasVehicles": vehicle_detected.get(feed['id'], False),
                "locationSource": feed.get('locationSource', '')
            }
        }
        map_features.append(feature)

    return {
        "type": "FeatureCollection",
        "features": map_features
    }


@app.get("/api/stream/config")
async def get_stream_config():
    """Get stream out configuration"""
    return app_state.stream_config.to_dict()


@app.post("/api/stream/config", response_model=StreamConfigResponse)
async def set_stream_config(config: StreamConfigRequest):
    """Set stream out configuration with audit logging"""
    # Update config via app_state (includes audit logging)
    updated_config = app_state.update_stream_config(
        enabled=config.enabled,
        format=config.format,
        ip=config.ip,
        port=config.port,
        latticeToken=config.latticeToken or "",
        latticeSandboxToken=config.latticeSandboxToken or "",
        latticeIntegration=config.latticeIntegration or "taiwan-cctv",
        latticeUrl=config.latticeUrl or "",
        chatsurferSession=config.chatsurferSession or "",
        chatsurferRoom=config.chatsurferRoom or "",
        chatsurferNickname=config.chatsurferNickname or "CCTV_Bot",
        chatsurferDomain=config.chatsurferDomain or "chatsurferxmppunclass"
    )

    # Initialize Lattice client if token provided and format is lattice
    if config.format == "lattice" and config.latticeToken:
        initialize_lattice_client(
            env_token=config.latticeToken,
            sandbox_token=config.latticeSandboxToken or None,
            base_url=config.latticeUrl or None
        )

    status = "enabled" if config.enabled else "disabled"
    if config.format == "cot":
        logger.info(f"Stream Out {status}: CoT", ip=config.ip, port=config.port)
    elif config.format == "lattice":
        logger.info(f"Stream Out {status}: Lattice", integration=config.latticeIntegration, url=config.latticeUrl or 'default')
    elif config.format == "chatsurfer":
        logger.info(f"Stream Out {status}: ChatSurfer", room=config.chatsurferRoom, nickname=config.chatsurferNickname)

    return {"status": "success", "config": updated_config.to_dict()}


@app.post("/api/database/reset")
async def reset_database():
    """Reset database - clear all detections and tracks"""
    if not app_state.db_manager:
        raise HTTPException(status_code=503, detail="Database not enabled")

    try:
        from database import Detection as DetectionModel, VehicleTrack as VehicleTrackModel
        from sqlalchemy import delete

        async with app_state.db_manager.session() as session:
            # Delete all vehicle tracks
            await session.execute(delete(VehicleTrackModel))
            # Delete all detections
            await session.execute(delete(DetectionModel))
            await session.commit()

        logger.info("Database reset completed - all detections and tracks cleared")

        # Audit log the reset
        app_state._log_audit("database_reset", {
            "action": "manual_reset",
            "timestamp": datetime.now(timezone.utc).isoformat()
        })

        await alert_manager.send_alert(
            alert_type="database_reset",
            severity="warning",
            message="Database has been manually reset",
            metadata={"timestamp": datetime.now(timezone.utc).isoformat()}
        )

        return {
            "status": "success",
            "message": "Database reset completed",
            "cleared": ["detections", "vehicle_tracks"]
        }

    except Exception as e:
        logger.error("Failed to reset database", error=str(e))
        raise HTTPException(status_code=500, detail=f"Database reset failed: {str(e)}")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time updates.

    Optional token-based authentication via query parameter:
    - ws://host:port/ws?token=YOUR_TOKEN&client_id=client123

    If CCTV_WS_AUTH_ENABLED=true, a valid token is required.
    """
    client_id = websocket.query_params.get("client_id", "anonymous")

    # Session-cookie auth. The HTTP middleware stack does not run for WebSocket
    # connections, so the same cookie is checked explicitly here.
    session_auth_enabled = os.environ.get("CCTV_AUTH_ENABLED", "true").lower() == "true"
    if session_auth_enabled and session_auth.get_users():
        cookie = websocket.cookies.get(session_auth.COOKIE_NAME)
        if not session_auth.verify_session_token(cookie):
            await websocket.close(code=4001, reason="Authentication required")
            return

    # Optional WebSocket authentication
    ws_auth_enabled = os.environ.get("CCTV_WS_AUTH_ENABLED", "false").lower() == "true"
    if ws_auth_enabled:
        token = websocket.query_params.get("token")
        valid_tokens = os.environ.get("CCTV_WS_TOKENS", "").split(",")
        valid_tokens = [t.strip() for t in valid_tokens if t.strip()]

        if valid_tokens and (not token or token not in valid_tokens):
            await websocket.close(code=4001, reason="Authentication required")
            logger.warning("WebSocket authentication failed", client_id=client_id)
            return

    if not app_state.ws_manager:
        await websocket.close(code=1011, reason="WebSocket manager not initialized")
        return

    await app_state.ws_manager.connect(websocket, client_id)
    logger.debug("WebSocket client connected", client_id=client_id)

    try:
        # Handle incoming messages
        await handle_websocket_messages(websocket, app_state.ws_manager)
    except Exception as e:
        logger.error("WebSocket error", client_id=client_id, error=str(e))
    finally:
        app_state.ws_manager.disconnect(websocket)
        logger.debug("WebSocket client disconnected", client_id=client_id)


# ============================================================================
# Static client
# ============================================================================
# Mounted last so every API route above still wins. Serving the UI from the same
# process (and therefore the same origin) is what lets the session cookie and the
# WebSocket work behind a single HTTPS hostname.
_CLIENT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "client")
if os.path.isdir(_CLIENT_DIR):
    app.mount("/", StaticFiles(directory=_CLIENT_DIR, html=True), name="client")
else:
    logger.warning("Client directory not found, UI will not be served", path=_CLIENT_DIR)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="info")
