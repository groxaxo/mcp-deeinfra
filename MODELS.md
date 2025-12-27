# Dynamic Model Support

The MCP DeepInfra server now features **real-time model discovery**, enabling you to dynamically access all available models from DeepInfra without manual configuration.

## Overview

The server automatically fetches and caches the list of available models from the DeepInfra API, allowing you to:

- 🔄 Access all current models in real-time
- 🚀 Discover new models as they become available
- ⚡ Benefit from intelligent caching (1-hour TTL)
- 🔍 Query model metadata and capabilities

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

Once you've discovered available models, you can configure any tool to use them via environment variables:

### Example: Using a Discovered Model

1. **Discover available models**:
   ```python
   models = list_models()
   # Find a model you want to use, e.g., "mistralai/Mixtral-8x7B-Instruct-v0.1"
   ```

2. **Configure the tool** in your `.env` file:
   ```bash
   MODEL_TEXT_GENERATION=mistralai/Mixtral-8x7B-Instruct-v0.1
   ```

3. **Use the tool** with your chosen model:
   ```python
   text_generation(prompt="Hello, how are you?")
   # This will now use Mixtral-8x7B-Instruct
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
- BGE models
- E5 models
- Custom embedding models

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

### With Claude Desktop

```json
{
  "mcpServers": {
    "deepinfra": {
      "command": "uv",
      "args": ["run", "mcp_deepinfra"],
      "env": {
        "DEEPINFRA_API_KEY": "your_api_key_here",
        "MODEL_TEXT_GENERATION": "meta-llama/Meta-Llama-3.3-70B-Instruct"
      }
    }
  }
}
```

First, use `list_models()` to discover available models, then configure your preferred model via the `MODEL_*` environment variables.

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
