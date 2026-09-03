"""
Integration tests for API endpoints
"""
import pytest


@pytest.mark.asyncio
class TestRootEndpoint:
    """Tests for root endpoint"""

    async def test_api_root_returns_info(self, async_client):
        """Test API root endpoint returns API info"""
        response = await async_client.get("/api")
        assert response.status_code == 200
        data = response.json()
        assert "name" in data

    async def test_root_redirects_to_ui(self, async_client):
        """Browsers hitting / should land on the UI, not the API JSON"""
        response = await async_client.get("/", follow_redirects=False)
        assert response.status_code == 302
        assert response.headers["location"] == "/index.html"


@pytest.mark.asyncio  
class TestHealthEndpoints:
    """Tests for health check endpoints"""

    async def test_liveness_probe(self, async_client):
        """Test liveness probe endpoint"""
        response = await async_client.get("/health/live")
        assert response.status_code == 200


@pytest.mark.asyncio
class TestFeedsEndpoint:
    """Tests for feeds endpoint"""

    async def test_get_feeds(self, async_client):
        """Test getting all feeds"""
        response = await async_client.get("/api/feeds")
        assert response.status_code == 200
        data = response.json()
        assert "feeds" in data


@pytest.mark.asyncio
class TestStatsEndpoint:
    """Tests for stats endpoint"""

    async def test_get_stats(self, async_client):
        """Test getting system statistics"""
        response = await async_client.get("/api/stats")
        assert response.status_code == 200


@pytest.mark.asyncio
class TestSearchEndpoint:
    """Tests for search endpoint"""

    async def test_search_no_params(self, async_client):
        """Test search without parameters"""
        response = await async_client.get("/api/search")
        assert response.status_code == 200
        data = response.json()
        assert "query" in data


@pytest.mark.asyncio
class TestMapEndpoint:
    """Tests for map data endpoint"""

    async def test_get_map_data(self, async_client):
        """Test getting GeoJSON map data"""
        response = await async_client.get("/api/map")
        assert response.status_code == 200
        data = response.json()
        assert data["type"] == "FeatureCollection"


@pytest.mark.asyncio
class TestStreamConfigEndpoint:
    """Tests for stream configuration endpoint"""

    async def test_get_stream_config(self, async_client):
        """Test getting stream configuration"""
        response = await async_client.get("/api/stream/config")
        assert response.status_code == 200


@pytest.mark.asyncio
class TestSnapshotEndpoint:
    """Tests for snapshot endpoint"""

    async def test_snapshot_not_found(self, async_client):
        """Test getting snapshot for non-existent feed"""
        response = await async_client.get("/api/feeds/non-existent/snapshot")
        assert response.status_code == 404



@pytest.mark.asyncio
class TestFeedsPayloadShape:
    """The feeds payload is hand-assembled for speed; guard its shape."""

    async def test_feeds_payload_is_valid_json_with_expected_keys(self, async_client):
        response = await async_client.get("/api/feeds")
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("application/json")
        data = response.json()
        assert set(data) == {"feeds", "status", "vehicleDetected", "lastUpdate"}
        assert isinstance(data["feeds"], list)
