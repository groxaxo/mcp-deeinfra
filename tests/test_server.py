"""Tests for MCP DeepInfra server initialization and tool listing."""

import pytest


class TestServerInitialization:
    """Test server initialization and basic functionality."""

    def test_server_initialization(self, mcp_server):
        """Test that the server initializes correctly."""
        # Server is already initialized in the fixture
        # If we get here, initialization was successful
        assert mcp_server is not None

    def test_tools_list(self, mcp_server):
        """Test that the server provides the expected tools."""
        response = mcp_server.send_request("tools/list")

        assert "result" in response
        assert "tools" in response["result"]

        tools = response["result"]["tools"]
        assert isinstance(tools, list)
        assert len(tools) > 0

        # Check that all expected tools are present
        tool_names = {tool["name"] for tool in tools}
        expected_tools = {
            "list_models",
            "generate_image",
            "text_generation",
            "embeddings",
            "reranker",
            "speech_recognition",
            "zero_shot_image_classification",
            "object_detection",
            "image_classification",
            "text_classification",
            "token_classification",
            "fill_mask"
        }

        assert expected_tools.issubset(tool_names), f"Missing tools: {expected_tools - tool_names}"

        # Verify each tool has required fields
        for tool in tools:
            assert "name" in tool
            assert "description" in tool
            assert isinstance(tool["description"], str)
            assert len(tool["description"]) > 0