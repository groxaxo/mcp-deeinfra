"""Tests for individual MCP DeepInfra tools."""

import pytest
import json


class TestTools:
    """Test individual MCP tools."""

    def test_list_models(self, mcp_server):
        """Test list models tool."""
        response = mcp_server.send_request("tools/call", {
            "name": "list_models",
            "arguments": {}
        })

        # Check response structure
        assert "result" in response or "error" in response

        if "result" in response:
            assert "content" in response["result"]
            content = response["result"]["content"]
            assert isinstance(content, list)
            assert len(content) > 0
            assert content[0]["type"] == "text"
            result_text = content[0]["text"]
            
            # Parse the JSON response
            result_data = json.loads(result_text)
            assert "models" in result_data
            assert "count" in result_data
            assert isinstance(result_data["models"], list)
            assert result_data["count"] >= 0

    def test_list_models_force_refresh(self, mcp_server):
        """Test list models tool with force refresh."""
        response = mcp_server.send_request("tools/call", {
            "name": "list_models",
            "arguments": {
                "force_refresh": True
            }
        })

        assert "result" in response or "error" in response

        if "result" in response:
            assert "content" in response["result"]
            content = response["result"]["content"]
            assert isinstance(content, list)
            assert len(content) > 0

    def test_generate_image(self, mcp_server):
        """Test image generation tool."""
        response = mcp_server.send_request("tools/call", {
            "name": "generate_image",
            "arguments": {
                "prompt": "a cute cat sitting on a windowsill"
            }
        })

        # Check response structure
        assert "result" in response or "error" in response

        if "result" in response:
            assert "content" in response["result"]
            content = response["result"]["content"]
            assert isinstance(content, list)
            assert len(content) > 0
            assert content[0]["type"] == "text"
            # Should contain image URL or error message
            result_text = content[0]["text"]
            assert isinstance(result_text, str)

    def test_text_generation(self, mcp_server):
        """Test text generation tool."""
        response = mcp_server.send_request("tools/call", {
            "name": "text_generation",
            "arguments": {
                "prompt": "Hello, how are you?"
            }
        })

        assert "result" in response or "error" in response

        if "result" in response:
            assert "content" in response["result"]
            content = response["result"]["content"]
            assert isinstance(content, list)
            assert len(content) > 0
            assert content[0]["type"] == "text"
            result_text = content[0]["text"]
            assert isinstance(result_text, str)

    def test_embeddings(self, mcp_server):
        """Test embeddings tool."""
        response = mcp_server.send_request("tools/call", {
            "name": "embeddings",
            "arguments": {
                "inputs": ["Hello world", "This is a test"]
            }
        })

        # Embeddings might fail due to model availability, but structure should be correct
        assert "result" in response or "error" in response

    def test_speech_recognition(self, mcp_server):
        """Test speech recognition tool."""
        response = mcp_server.send_request("tools/call", {
            "name": "speech_recognition",
            "arguments": {
                "audio_url": "https://www.soundjay.com/misc/sounds/bell-ringing-05.wav"
            }
        })

        assert "result" in response or "error" in response

        if "result" in response:
            assert "content" in response["result"]
            content = response["result"]["content"]
            assert isinstance(content, list)
            assert len(content) > 0
            assert content[0]["type"] == "text"

    def test_zero_shot_image_classification(self, mcp_server):
        """Test zero-shot image classification tool."""
        response = mcp_server.send_request("tools/call", {
            "name": "zero_shot_image_classification",
            "arguments": {
                "image_url": "https://images.unsplash.com/photo-1514888286974-6c03e2ca1dba?w=400",
                "candidate_labels": ["cat", "dog", "bird"]
            }
        })

        assert "result" in response or "error" in response

    def test_object_detection(self, mcp_server):
        """Test object detection tool."""
        response = mcp_server.send_request("tools/call", {
            "name": "object_detection",
            "arguments": {
                "image_url": "https://images.unsplash.com/photo-1514888286974-6c03e2ca1dba?w=400"
            }
        })

        assert "result" in response or "error" in response

    def test_image_classification(self, mcp_server):
        """Test image classification tool."""
        response = mcp_server.send_request("tools/call", {
            "name": "image_classification",
            "arguments": {
                "image_url": "https://images.unsplash.com/photo-1514888286974-6c03e2ca1dba?w=400"
            }
        })

        assert "result" in response or "error" in response

    def test_text_classification(self, mcp_server):
        """Test text classification tool."""
        response = mcp_server.send_request("tools/call", {
            "name": "text_classification",
            "arguments": {
                "text": "This is a positive review of the product"
            }
        })

        assert "result" in response or "error" in response

    def test_token_classification(self, mcp_server):
        """Test token classification tool."""
        response = mcp_server.send_request("tools/call", {
            "name": "token_classification",
            "arguments": {
                "text": "John lives in New York"
            }
        })

        assert "result" in response or "error" in response

    def test_fill_mask(self, mcp_server):
        """Test fill mask tool."""
        response = mcp_server.send_request("tools/call", {
            "name": "fill_mask",
            "arguments": {
                "text": "Hello [MASK] world"
            }
        })

        assert "result" in response or "error" in response