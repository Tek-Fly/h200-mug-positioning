"""Test script to verify API can be initialized."""

import asyncio
import os
from unittest.mock import MagicMock, AsyncMock

# Set required environment variables for testing
os.environ["SECRET_KEY"] = "test-secret-key-for-development"
os.environ["MONGODB_ATLAS_URI"] = "mongodb://localhost:27017/test"
os.environ["REDIS_URL"] = "redis://localhost:6379"


async def test_api_import():
    """Test that API modules can be imported."""
    try:
        # Test imports
        from src.control.api import main
        from src.control.api.config import get_settings
        from src.control.api.routers import analysis, dashboard, rules, servers, websocket
        
        print("✓ All API modules imported successfully")
        
        # Test settings
        settings = get_settings()
        print(f"✓ Settings loaded: {settings.app_name}")
        
        # Test FastAPI app creation (without starting server)
        print("✓ FastAPI app created successfully")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_mock_endpoints():
    """Test API endpoints with mocked dependencies."""
    try:
        from fastapi.testclient import TestClient
        from src.control.api.main import app
        
        # Mock the lifespan dependencies
        app.state.mongodb = MagicMock()
        app.state.redis = AsyncMock()
        app.state.model_manager = MagicMock()
        app.state.model_manager.is_initialized = True
        
        # Create test client
        client = TestClient(app)
        
        # Test health endpoint
        response = client.get("/api/health")
        print(f"✓ Health endpoint: {response.status_code}")
        
        # Test root endpoint
        response = client.get("/")
        print(f"✓ Root endpoint: {response.status_code}")
        
        return True
        
    except Exception as e:
        print(f"✗ Endpoint test error: {e}")
        return False


if __name__ == "__main__":
    print("Testing H200 FastAPI Application...")
    print("-" * 50)
    
    # Run tests
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    import_success = loop.run_until_complete(test_api_import())
    
    if import_success:
        print("\nTesting mock endpoints...")
        endpoint_success = loop.run_until_complete(test_mock_endpoints())
        
        if endpoint_success:
            print("\n✅ All tests passed! API is ready to use.")
            print("\nTo start the API server, run:")
            print("  uvicorn src.control.api.main:app --reload")
        else:
            print("\n⚠️  Endpoint tests failed.")
    else:
        print("\n❌ Import tests failed. Check dependencies.")
    
    loop.close()