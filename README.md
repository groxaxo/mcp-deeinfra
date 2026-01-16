# MCP DeepInfra AI Tools Server

A powerful Model Context Protocol (MCP) server that provides comprehensive AI capabilities using the DeepInfra OpenAI-compatible API. This server features **real-time model discovery** and **automatic model updates**, allowing you to dynamically access all available models from DeepInfra without manual configuration.

## Features

✨ **Real-Time Model Access** - Automatically discover and use all available DeepInfra models
🚀 **Auto-Updated Models** - Models are automatically updated server-side by DeepInfra
🏠 **Self-Hostable** - Run anywhere with Python 3.10+ and a DeepInfra API key
🎨 **Image Generation** - Create stunning images from text prompts
📝 **Text Processing** - Advanced text generation and completion
🔤 **Embeddings** - Generate vector embeddings for semantic search
🔄 **Reranking** - Rerank documents by relevance using state-of-the-art reranker models
🎙️ **Speech Recognition** - Transcribe audio using Whisper models
🔍 **Computer Vision** - Image classification, object detection, and zero-shot classification
🏷️ **Text Analysis** - Sentiment analysis, NER, and text classification
🎭 **Fill Mask** - Context-aware word prediction

## Self-Hosting and Auto-Updates

This MCP server is designed to be **fully self-hostable** and requires only:
- Python 3.10 or higher
- A DeepInfra API key (free tier available)
- Network access to DeepInfra's API

**Auto-Updates:** All models are hosted and updated by DeepInfra server-side. When DeepInfra adds or updates models:
- New models become immediately available through the API
- The `list_models` tool discovers them automatically (with 1-hour cache)
- No manual updates or downloads required
- Use `force_refresh=True` to immediately discover new models

This ensures you always have access to the latest models without any maintenance overhead.

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

## Quick Start

1. **Install uv** (if not already installed):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **Clone this repository**:
   ```bash
   git clone https://github.com/groxaxo/mcp-deeinfra.git
   cd mcp-deeinfra
   ```

3. **Install dependencies**:
   ```bash
   uv sync
   ```

4. **Set up your DeepInfra API key**:
   
   Copy the example configuration and add your API key:
   ```bash
   cp .env.example .env
   # Then edit .env and replace 'your_api_key_here' with your actual API key
   ```
   
   Or create a `.env` file directly:
   ```bash
   echo "DEEPINFRA_API_KEY=your_api_key_here" > .env
   ```
   
   That's it! The server comes pre-configured with sensible default models for all tools.
   
   **Get your free API key:** Sign up at [DeepInfra](https://deepinfra.com/)

## Configuration

**All tools work out of the box** with pre-configured default models. You only need to set your `DEEPINFRA_API_KEY`.

### Advanced Configuration (Optional)

If you want to customize the default models or enabled tools, you can add these environment variables to your `.env` file:

- `ENABLED_TOOLS`: Comma-separated list of tools to enable (default: "all")
  - Example: `ENABLED_TOOLS=generate_image,text_generation,embeddings`

- `MODEL_GENERATE_IMAGE`: Default model for image generation (default: "black-forest-labs/FLUX-1-dev")

- `MODEL_TEXT_GENERATION`: Default model for text generation (default: "meta-llama/Meta-Llama-3.3-70B-Instruct")

- `MODEL_EMBEDDINGS`: Default model for embeddings (default: "BAAI/bge-large-en-v1.5")

- `MODEL_RERANKER`: Default model for reranking (default: "Qwen/Qwen3-Reranker-4B")

- `MODEL_SPEECH_RECOGNITION`: Default model for speech recognition (default: "openai/whisper-large-v3")

- `MODEL_ZERO_SHOT_IMAGE_CLASSIFICATION`: Default model for zero-shot image classification (default: "meta-llama/Llama-3.2-90B-Vision-Instruct")

- `MODEL_OBJECT_DETECTION`: Default model for object detection (default: "meta-llama/Llama-3.2-90B-Vision-Instruct")

- `MODEL_IMAGE_CLASSIFICATION`: Default model for image classification (default: "meta-llama/Llama-3.2-90B-Vision-Instruct")

- `MODEL_TEXT_CLASSIFICATION`: Default model for text classification (default: "meta-llama/Meta-Llama-3.3-70B-Instruct")

- `MODEL_TOKEN_CLASSIFICATION`: Default model for token classification (default: "meta-llama/Meta-Llama-3.3-70B-Instruct")

- `MODEL_FILL_MASK`: Default model for fill mask (default: "meta-llama/Meta-Llama-3.3-70B-Instruct")

You can discover available models using the `list_models` tool and configure any compatible model from DeepInfra's catalog.

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

### Claude Desktop Setup

Add this to your `claude_desktop_config.json` (typically located in `~/Library/Application Support/Claude/` on macOS or `%APPDATA%\Claude\` on Windows):

```json
{
  "mcpServers": {
    "deepinfra": {
      "command": "uv",
      "args": ["--directory", "/path/to/mcp-deeinfra", "run", "mcp_deepinfra"],
      "env": {
        "DEEPINFRA_API_KEY": "your_api_key_here"
      }
    }
  }
}
```

**Important:** Replace `/path/to/mcp-deeinfra` with the actual path where you cloned this repository.

**Getting your API key:** Sign up at [DeepInfra](https://deepinfra.com/) to get your free API key.

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

### Reranking
- **`reranker`**: Rerank a list of documents based on their relevance to a query. Returns ranked results with relevance scores.

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

All tools come pre-configured with optimal default models. You can optionally customize models by setting environment variables (see Advanced Configuration section above).

For detailed information about model discovery and customization, see [MODELS.md](MODELS.md).

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