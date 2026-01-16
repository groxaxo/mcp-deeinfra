# Dynamic Model Support

The MCP DeepInfra server now features **real-time model discovery** and **automatic updates**, enabling you to dynamically access all available models from DeepInfra without manual configuration.

## Overview

The server automatically fetches and caches the list of available models from the DeepInfra API, allowing you to:

- 🔄 Access all current models in real-time with automatic server-side updates
- 🚀 Discover new models as they become available on DeepInfra
- ⚡ Benefit from intelligent caching (1-hour TTL)
- 🔍 Query model metadata and capabilities
- 🏠 Self-host with zero maintenance for model updates

## Self-Hosting and Auto-Updates

**Self-Hostable:** This server can run anywhere with:
- Python 3.10 or higher
- A DeepInfra API key
- Internet access to DeepInfra's API endpoints

**Auto-Updates:** DeepInfra handles all model updates server-side:
- New models are immediately available via API
- Model improvements and updates are automatic
- No manual intervention or downloads required
- Use `list_models(force_refresh=True)` to bypass cache and get the latest model list

## Using the list_models Tool

The `list_models` tool provides access to DeepInfra's complete model catalog.

### Basic Usage

```python
# List all available models (uses cache if available)
list_models()
```

### Force Refresh

```python
# Bypass cache and fetch fresh model list
list_models(force_refresh=True)
```

### Response Format

The tool returns a JSON object with the following structure:

```json
{
  "models": [
    {
      "id": "meta-llama/Meta-Llama-3.3-70B-Instruct",
      "created": 1234567890,
      "owned_by": "deepinfra"
    },
    {
      "id": "black-forest-labs/FLUX-1-dev",
      "created": 1234567890,
      "owned_by": "deepinfra"
    }
    // ... more models
  ],
  "count": 150,
  "cached": true,
  "cache_age_seconds": 1800
}
```

### Response Fields

- **models**: Array of model objects, each containing:
  - `id`: The model identifier (used in API calls)
  - `created`: Unix timestamp when model was added
  - `owned_by`: Model owner/provider
- **count**: Total number of available models
- **cached**: Whether this response actually came from cache (accounts for fallback scenarios)
- **cache_age_seconds**: Age of cached data (0 if fresh)

## Caching Mechanism

To optimize performance and reduce API calls, the server implements intelligent caching:

- **Cache Duration**: 1 hour (3600 seconds)
- **Automatic Refresh**: Cache is automatically refreshed when expired
- **Manual Refresh**: Use `force_refresh=True` to bypass cache
- **Fallback**: If API call fails, returns cached data if available

## Using Models with Tools

**All tools come with pre-configured default models**, so you can use them immediately without any additional configuration.

If you want to use a different model, you can customize it via environment variables:

### Example: Customizing the Text Generation Model (Optional)

1. **Discover available models**:
   ```python
   models = list_models()
   # Find a model you want to use, e.g., "mistralai/Mixtral-8x7B-Instruct-v0.1"
   ```

2. **Configure the tool** in your `.env` file:
   ```bash
   DEEPINFRA_API_KEY=your_api_key_here
   MODEL_TEXT_GENERATION=mistralai/Mixtral-8x7B-Instruct-v0.1
   ```

3. **Use the tool** with your chosen model:
   ```python
   text_generation(prompt="Hello, how are you?")
   # This will now use Mixtral-8x7B-Instruct instead of the default
   ```

### Example: Customizing the Reranker Model (Optional)

The reranker tool comes pre-configured with Qwen/Qwen3-Reranker-4B. To use it:

```python
reranker(
    query="What is the capital of France?",
    documents=[
        "Paris is the capital of France.",
        "London is the capital of the United Kingdom.",
        "Berlin is the capital of Germany."
    ],
    top_n=3
)
# Returns documents ranked by relevance to the query
```

If you want to use a different reranker model:

1. **Discover available reranker models**:
   ```python
   models = list_models()
   # Find reranker models like "Qwen/Qwen3-Reranker-8B"
   ```

2. **Configure the reranker** in your `.env` file:
   ```bash
   DEEPINFRA_API_KEY=your_api_key_here
   MODEL_RERANKER=Qwen/Qwen3-Reranker-8B
   ```

3. **Use the reranker** as usual:
   ```python
   reranker(
       query="What is the capital of France?",
       documents=[
           "Paris is the capital of France.",
           "London is the capital of the United Kingdom.",
           "Berlin is the capital of Germany."
       ],
       top_n=3
   )
   # Returns documents ranked by relevance to the query
   ```

## Model Categories

DeepInfra provides models across various categories:

### Text Generation
- LLaMA models (Meta)
- Mixtral models (Mistral AI)
- Qwen models
- And many more...

### Image Generation
- FLUX models (Black Forest Labs)
- Stable Diffusion variants
- And more...

### Embeddings
- BAAI models (bge-large, bge-base)
- E5 models
- Custom embedding models

### Rerankers
- Qwen3-Reranker-0.6B (lightweight, efficient)
- Qwen3-Reranker-4B (balanced performance)
- Qwen3-Reranker-8B (best-in-class accuracy)

### Vision Models
- LLaMA Vision
- CLIP variants
- Multimodal models

### Audio Processing
- Whisper (OpenAI)
- Speech recognition models

## Best Practices

1. **Cache Utilization**: Use the default cached response for most queries to minimize API calls
2. **Periodic Refresh**: Force refresh periodically (e.g., daily) to discover new models
3. **Model Selection**: Choose models based on your specific use case and performance requirements
4. **Environment Variables**: Configure default models via environment variables for consistent behavior

## API Endpoint

The server uses DeepInfra's OpenAI-compatible endpoint:

```
GET https://api.deepinfra.com/v1/openai/models
```

This endpoint provides the complete list of available models with their metadata.

## Error Handling

The server gracefully handles various error scenarios:

- **API Unavailable**: Returns cached data if available
- **Network Errors**: Falls back to cache
- **Invalid Response**: Returns empty list with error message
- **Missing API Key**: Raises clear error message

## Integration Examples

### With Claude Desktop - Minimal Configuration

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

This minimal configuration is all you need - all tools will use their pre-configured default models.

### With Claude Desktop - Custom Models (Optional)

If you want to override default models:

```json
{
  "mcpServers": {
    "deepinfra": {
      "command": "uv",
      "args": ["--directory", "/path/to/mcp-deeinfra", "run", "mcp_deepinfra"],
      "env": {
        "DEEPINFRA_API_KEY": "your_api_key_here",
        "MODEL_TEXT_GENERATION": "mistralai/Mixtral-8x7B-Instruct-v0.1"
      }
    }
  }
}
```

Use `list_models()` to discover available models, then configure your preferred models via the `MODEL_*` environment variables as needed.

## Troubleshooting

### Models Not Loading

1. Check your API key is valid
2. Verify network connectivity
3. Try force refresh: `list_models(force_refresh=True)`

### Cache Not Updating

- The cache updates automatically after 1 hour
- Use `force_refresh=True` to manually refresh

### Specific Model Not Available

- Use `list_models()` to see current available models
- Model availability may change based on DeepInfra's offerings
- Check the DeepInfra documentation for model status

## Additional Resources

- [DeepInfra API Documentation](https://deepinfra.com/docs/api-reference)
- [DeepInfra Models Page](https://deepinfra.com/docs/models)
- [OpenAI API Compatibility](https://deepinfra.com/docs/api-reference)
