"""
Test suite for health check controller.

Tests cover:
- Root endpoint
- Health check endpoint
- Status responses
"""

import pytest
from unittest.mock import patch

from src.api.controllers.health_controller import root, health


# ============================================================================
# Root Endpoint Tests
# ============================================================================

class TestRootEndpoint:
    """Test the root endpoint"""
    
    @pytest.mark.asyncio
    async def test_root_returns_server_info(self):
        """Test that root endpoint returns server information"""
        response = await root()
        
        assert isinstance(response, dict)
        assert "message" in response
        assert "model" in response
        assert "version" in response
        
        assert response["message"] == "LLM Server is running"
        assert response["version"] == "1.0.0"
        
        print("✓ Root endpoint returns server info")
    
    @pytest.mark.asyncio
    async def test_root_includes_model_path(self):
        """Test that root endpoint includes model path from settings"""
        with patch('src.api.controllers.health_controller.settings') as mock_settings:
            mock_settings.model_path = "/test/model.gguf"
            
            response = await root()
            
            assert response["model"] == "/test/model.gguf"
        
        print("✓ Root endpoint includes model path")
    
    @pytest.mark.asyncio
    async def test_root_response_structure(self):
        """Test root endpoint response structure"""
        response = await root()
        
        # Verify all required fields are present
        required_fields = ["message", "model", "version"]
        for field in required_fields:
            assert field in response
        
        # Verify data types
        assert isinstance(response["message"], str)
        assert isinstance(response["model"], str)
        assert isinstance(response["version"], str)
        
        print("✓ Root endpoint has correct response structure")


# ============================================================================
# Health Check Endpoint Tests
# ============================================================================

class TestHealthEndpoint:
    """Test the health check endpoint"""
    
    @pytest.mark.asyncio
    async def test_health_check_returns_healthy_status(self):
        """Test that health check returns healthy status"""
        response = await health()
        
        assert isinstance(response, dict)
        assert "status" in response
        assert response["status"] == "healthy"
        
        print("✓ Health check returns healthy status")
    
    @pytest.mark.asyncio
    async def test_health_check_response_format(self):
        """Test health check response format"""
        response = await health()
        
        # Should have exactly one field
        assert len(response) == 1
        assert "status" in response
        
        # Status should be a string
        assert isinstance(response["status"], str)
        
        print("✓ Health check has correct response format")


# ============================================================================
# Integration Tests
# ============================================================================

class TestHealthIntegration:
    """Integration tests for health endpoints"""
    
    @pytest.mark.asyncio
    async def test_both_endpoints_are_accessible(self):
        """Test that both health endpoints work"""
        root_response = await root()
        health_response = await health()
        
        assert root_response is not None
        assert health_response is not None
        
        print("✓ Both health endpoints are accessible")
    
    @pytest.mark.asyncio
    async def test_endpoints_return_consistent_data(self):
        """Test that endpoints return consistent data"""
        # Call multiple times
        response1 = await health()
        response2 = await health()
        response3 = await health()
        
        # All should return the same status
        assert response1 == response2 == response3
        assert all(r["status"] == "healthy" for r in [response1, response2, response3])
        
        print("✓ Health endpoints return consistent data")


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    import asyncio
    
    print("\n" + "="*70)
    print("Running Health Controller Tests")
    print("="*70 + "\n")
    
    # Test root endpoint
    print("--- Root Endpoint Tests ---")
    root_tests = TestRootEndpoint()
    asyncio.run(root_tests.test_root_returns_server_info())
    asyncio.run(root_tests.test_root_includes_model_path())
    asyncio.run(root_tests.test_root_response_structure())
    
    # Test health check endpoint
    print("\n--- Health Check Endpoint Tests ---")
    health_tests = TestHealthEndpoint()
    asyncio.run(health_tests.test_health_check_returns_healthy_status())
    asyncio.run(health_tests.test_health_check_response_format())
    
    # Test integration
    print("\n--- Integration Tests ---")
    integration_tests = TestHealthIntegration()
    asyncio.run(integration_tests.test_both_endpoints_are_accessible())
    asyncio.run(integration_tests.test_endpoints_return_consistent_data())
    
    print("\n" + "="*70)
    print("All Health Controller Tests Passed! ✓")
    print("="*70 + "\n")
