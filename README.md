# MCP DeepInfra AI Tools Server

A powerful Model Context Protocol (MCP) server that provides comprehensive AI capabilities using the DeepInfra OpenAI-compatible API. This server features **real-time model discovery**, allowing you to dynamically access all available models from DeepInfra without manual configuration.

## Features

✨ **Real-Time Model Access** - Automatically discover and use all available DeepInfra models
🎨 **Image Generation** - Create stunning images from text prompts
📝 **Text Processing** - Advanced text generation and completion
🔤 **Embeddings** - Generate vector embeddings for semantic search
🎙️ **Speech Recognition** - Transcribe audio using Whisper models
🔍 **Computer Vision** - Image classification, object detection, and zero-shot classification
🏷️ **Text Analysis** - Sentiment analysis, NER, and text classification
🎭 **Fill Mask** - Context-aware word prediction

## Acknowledgments

This project builds upon the foundation of the DeepInfra API integration and extends it with dynamic model discovery capabilities. Special thanks to **Vlad J** and all contributors who have helped shape this project.

## Project Structure

```
mcp-deepinfra/
├── src/
│   └── mcp_deepinfra/
│       ├── __init__.py      # Package initialization
│       └── server.py        # Main MCP server implementation with dynamic model discovery
├── tests/
│   ├── conftest.py          # Pytest fixtures and configuration
│   ├── test_server.py       # Server initialization tests
│   └── test_tools.py        # Individual tool tests (including list_models)
├── pyproject.toml           # Project configuration and dependencies
├── uv.lock                  # Lock file for uv package manager
├── run_tests.sh             # Convenience script for running tests
├── MODELS.md                # Detailed documentation on dynamic model support
└── README.md                # This file
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

### Model Discovery
- **`list_models`**: Fetch all available models from DeepInfra in real-time. Supports caching with 1-hour TTL and optional force refresh. Returns comprehensive model information including IDs, ownership, and metadata.

### Image Generation
- **`generate_image`**: Generate an image from a text prompt. Returns the URL of the generated image.

### Text Processing
- **`text_generation`**: Generate text completion from a prompt.
- **`text_classification`**: Analyze text for sentiment and category.
- **`token_classification`**: Perform named entity recognition (NER) on text.
- **`fill_mask`**: Fill masked tokens in text with appropriate words.

### Embeddings
- **`embeddings`**: Generate embeddings for a list of input texts.

### Audio Processing
- **`speech_recognition`**: Transcribe audio from a URL to text using Whisper model.

### Computer Vision
- **`zero_shot_image_classification`**: Classify an image into provided candidate labels using vision model.
- **`object_detection`**: Detect and describe objects in an image using multimodal model.
- **`image_classification`**: Classify and describe contents of an image using multimodal model.

## Dynamic Model Support

The server now features **real-time model discovery**! Use the `list_models` tool to:
- Get an up-to-date list of all available DeepInfra models
- View model metadata and capabilities
- Discover new models as they become available
- Cache results for better performance (1-hour TTL by default)

Example usage:
```python
# List all available models (uses cache if available)
list_models()

# Force refresh to get the latest models
list_models(force_refresh=True)
```

All tools can leverage any compatible model from DeepInfra's extensive catalog by setting the appropriate environment variables.

For detailed information about model discovery and usage, see [MODELS.md](MODELS.md).

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