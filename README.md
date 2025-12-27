
# MCP DeepInfra AI Tools Server

This is a Model Context Protocol (MCP) server that provides various AI capabilities using the DeepInfra OpenAI-compatible API, including image generation, text processing, embeddings, speech recognition, and more.

## Project Structure

```
mcp-deepinfra/
├── src/
│   └── mcp_deepinfra/
│       ├── __init__.py      # Package initialization
│       └── server.py        # Main MCP server implementation
├── tests/
│   ├── conftest.py          # Pytest fixtures and configuration
│   ├── test_server.py      # Server initialization tests
│   └── test_tools.py        # Individual tool tests
├── pyproject.toml           # Project configuration and dependencies
├── uv.lock                  # Lock file for uv package manager
├── run_tests.sh             # Convenience script for running tests
└── README.md               # This file
```

## Setup

1. Install uv if not already installed:
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. Clone or download this repository.

3. Install dependencies:
   ```bash
   uv sync
   ```

4. Set up your DeepInfra API key:
   Create a `.env` file in the project root:
   ```
   DEEPINFRA_API_KEY=your_api_key_here
   ```

## Configuration

You can configure which tools are enabled and set default models for each tool using environment variables in your `.env` file:

- `ENABLED_TOOLS`: Comma-separated list of tools to enable. Use "all" to enable all tools (default: "all"). Example: `ENABLED_TOOLS=generate_image,text_generation,embeddings`

- `MODEL_GENERATE_IMAGE`: Default model for image generation (default: "black-forest-labs/FLUX-1-dev")

- `MODEL_TEXT_GENERATION`: Default model for text generation (default: "meta-llama/Meta-Llama-3.3-70B-Instruct")

- `MODEL_EMBEDDINGS`: Default model for embeddings (default: "BAAI/bge-large-en-v1.5")

- `MODEL_SPEECH_RECOGNITION`: Default model for speech recognition (default: "openai/whisper-large-v3")

- `MODEL_ZERO_SHOT_IMAGE_CLASSIFICATION`: Default model for zero-shot image classification (default: "meta-llama/Llama-3.2-90B-Vision-Instruct")

- `MODEL_OBJECT_DETECTION`: Default model for object detection (default: "meta-llama/Llama-3.2-90B-Vision-Instruct")

- `MODEL_IMAGE_CLASSIFICATION`: Default model for image classification (default: "meta-llama/Llama-3.2-90B-Vision-Instruct")

- `MODEL_TEXT_CLASSIFICATION`: Default model for text classification (default: "meta-llama/Meta-Llama-3.3-70B-Instruct")

- `MODEL_TOKEN_CLASSIFICATION`: Default model for token classification (default: "meta-llama/Meta-Llama-3.3-70B-Instruct")

- `MODEL_FILL_MASK`: Default model for fill mask (default: "meta-llama/Meta-Llama-3.3-70B-Instruct")

The tools always use the models specified via environment variables. Model selection is configured at startup time through the environment variables listed above.

## Running the Server

To run the server locally:
```bash
uv run mcp_deepinfra
```

Or directly with Python:
```bash
python -m mcp_deepinfra.server
```

## Using with MCP Clients

Configure your MCP client (e.g., Claude Desktop) to use this server.

For Claude Desktop, add to your `claude_desktop_config.json`:
```json
{
  "mcpServers": {
    "deepinfra": {
      "command": "uv",
      "args": ["run", "mcp_deepinfra"],
      "env": {
        "DEEPINFRA_API_KEY": "your_api_key_here"
      }
    }
  }
}
```

## Tools Provided

This server provides the following MCP tools:

- `generate_image`: Generate an image from a text prompt. Returns the URL of the generated image.
- `text_generation`: Generate text completion from a prompt.
- `embeddings`: Generate embeddings for a list of input texts.
- `speech_recognition`: Transcribe audio from a URL to text using Whisper model.
- `zero_shot_image_classification`: Classify an image into provided candidate labels using vision model.
- `object_detection`: Detect and describe objects in an image using multimodal model.
- `image_classification`: Classify and describe contents of an image using multimodal model.
- `text_classification`: Analyze text for sentiment and category.
- `token_classification`: Perform named entity recognition (NER) on text.
- `fill_mask`: Fill masked tokens in text with appropriate words.

## Testing

To test the server locally, run the pytest test suite:
```bash
# Install test dependencies
uv sync --extra test

# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test file
pytest tests/test_tools.py

# Use the convenience script
./run_tests.sh
```

The tests include:
- Server initialization and tool listing
- Individual tool functionality tests via JSON-RPC protocol
- All tests run synchronously without async/await complexity

## Running with uvx

`uvx` is designed for running published Python packages from PyPI or GitHub. For local development, use the `uv run` command as described above.

If you publish this package to PyPI (e.g., as `mcp-deepinfra`), you can run it with:
```bash
uvx mcp-deepinfra
```

And configure your MCP client to use:
```json
{
  "mcpServers": {
    "deepinfra": {
      "command": "uvx",
      "args": ["mcp-deepinfra"],
      "env": {
        "DEEPINFRA_API_KEY": "your_api_key_here"
      }
    }
  }
}
```

For local development, stick with the `uv run` approach.