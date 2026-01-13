import asyncio
from mcp.server.fastmcp import FastMCP
import httpx
import os
from dotenv import load_dotenv
import json
from openai import AsyncOpenAI
import time

load_dotenv()

DEEPINFRA_API_KEY = os.getenv("DEEPINFRA_API_KEY")

if not DEEPINFRA_API_KEY:
    raise ValueError("DEEPINFRA_API_KEY not set")

app = FastMCP("deepinfra-ai-tools")

# Initialize OpenAI client with DeepInfra base URL
client = AsyncOpenAI(
    api_key=DEEPINFRA_API_KEY,
    base_url="https://api.deepinfra.com/v1/openai"
)

# Cache for models list
_models_cache = None
_models_cache_timestamp = None
_models_cache_ttl = 3600  # 1 hour cache TTL

# Configuration
ENABLED_TOOLS_STR = os.getenv("ENABLED_TOOLS", "all")
if ENABLED_TOOLS_STR == "all":
    ENABLED_TOOLS = ["all"]
else:
    ENABLED_TOOLS = [tool.strip() for tool in ENABLED_TOOLS_STR.split(",")]

DEFAULT_MODELS = {
    "generate_image": os.getenv("MODEL_GENERATE_IMAGE", "black-forest-labs/FLUX-1-dev"),
    "text_generation": os.getenv("MODEL_TEXT_GENERATION", "meta-llama/Meta-Llama-3.3-70B-Instruct"),
    "embeddings": os.getenv("MODEL_EMBEDDINGS", "BAAI/bge-large-en-v1.5"),
    "reranker": os.getenv("MODEL_RERANKER", "Qwen/Qwen3-Reranker-4B"),
    "speech_recognition": os.getenv("MODEL_SPEECH_RECOGNITION", "openai/whisper-large-v3"),
    "zero_shot_image_classification": os.getenv("MODEL_ZERO_SHOT_IMAGE_CLASSIFICATION", "meta-llama/Llama-3.2-90B-Vision-Instruct"),
    "object_detection": os.getenv("MODEL_OBJECT_DETECTION", "meta-llama/Llama-3.2-90B-Vision-Instruct"),
    "image_classification": os.getenv("MODEL_IMAGE_CLASSIFICATION", "meta-llama/Llama-3.2-90B-Vision-Instruct"),
    "text_classification": os.getenv("MODEL_TEXT_CLASSIFICATION", "meta-llama/Meta-Llama-3.3-70B-Instruct"),
    "token_classification": os.getenv("MODEL_TOKEN_CLASSIFICATION", "meta-llama/Meta-Llama-3.3-70B-Instruct"),
    "fill_mask": os.getenv("MODEL_FILL_MASK", "meta-llama/Meta-Llama-3.3-70B-Instruct"),
}


async def get_available_models(force_refresh: bool = False) -> tuple[list[dict], bool]:
    """Fetch available models from DeepInfra API with caching.
    
    Returns:
        Tuple of (models_list, was_cached) where was_cached indicates if data came from cache.
    """
    global _models_cache, _models_cache_timestamp
    
    # Check if cache is valid
    current_time = time.time()
    if (not force_refresh and 
        _models_cache is not None and 
        _models_cache_timestamp is not None and
        (current_time - _models_cache_timestamp) < _models_cache_ttl):
        return _models_cache, True
    
    # Fetch fresh models list
    try:
        models_response = await client.models.list()
        models_list = []
        for model in models_response.data:
            models_list.append({
                "id": model.id,
                "created": getattr(model, "created", None),
                "owned_by": getattr(model, "owned_by", "deepinfra")
            })
        
        # Update cache
        _models_cache = models_list
        _models_cache_timestamp = current_time
        
        return models_list, False
    except Exception as e:
        # If fetch fails and we have cache, return cache
        # Invariant: if _models_cache is not None, _models_cache_timestamp must also be set
        # from a previous successful fetch (both are set together at lines 79-80)
        if _models_cache is not None:
            assert _models_cache_timestamp is not None, "Cache timestamp should be set when cache exists"
            return _models_cache, True
        # Otherwise, return empty list
        return [], False


if "all" in ENABLED_TOOLS or "list_models" in ENABLED_TOOLS:
    @app.tool()
    async def list_models(force_refresh: bool = False) -> str:
        """
        List all available models from DeepInfra API in real-time.
        
        Args:
            force_refresh: If True, bypass cache and fetch fresh model list. Default is False.
        
        Returns:
            JSON string containing list of available models with their IDs and metadata.
        """
        try:
            models, was_cached = await get_available_models(force_refresh)
            # Cache age is only meaningful when data comes from cache
            # Store timestamp to avoid potential race condition
            timestamp = _models_cache_timestamp
            cache_age = int(time.time() - timestamp) if (was_cached and timestamp) else 0
            return json.dumps({
                "models": models,
                "count": len(models),
                "cached": was_cached,
                "cache_age_seconds": cache_age
            }, indent=2)
        except Exception as e:
            return f"Error fetching models: {type(e).__name__}: {str(e)}"



if "all" in ENABLED_TOOLS or "generate_image" in ENABLED_TOOLS:
    @app.tool()
    async def generate_image(prompt: str) -> str:
        """Generate an image from a text prompt using DeepInfra OpenAI-compatible API."""
        model = DEFAULT_MODELS["generate_image"]
        try:
            response = await client.images.generate(
                model=model,
                prompt=prompt,
                n=1,
            )
            if response.data:
                return f"Generated image URL: {response.data[0].url}"
            else:
                return "No image generated"
        except Exception as e:
            return f"Error generating image: {type(e).__name__}: {str(e)}"

if "all" in ENABLED_TOOLS or "text_generation" in ENABLED_TOOLS:
    @app.tool()
    async def text_generation(prompt: str) -> str:
        """Generate text completion using DeepInfra OpenAI-compatible API."""
        model = DEFAULT_MODELS["text_generation"]
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=256,
                temperature=0.7,
            )
            if response.choices:
                return response.choices[0].message.content
            else:
                return "No text generated"
        except Exception as e:
            return f"Error generating text: {type(e).__name__}: {str(e)}"

if "all" in ENABLED_TOOLS or "embeddings" in ENABLED_TOOLS:
    @app.tool()
    async def embeddings(inputs: list[str]) -> str:
        """Generate embeddings for a list of texts using DeepInfra OpenAI-compatible API."""
        model = DEFAULT_MODELS["embeddings"]
        try:
            response = await client.embeddings.create(
                model=model,
                input=inputs,
            )
            embeddings_list = [item.embedding for item in response.data]
            return str(embeddings_list)
        except Exception as e:
            return f"Error generating embeddings: {type(e).__name__}: {str(e)}"

if "all" in ENABLED_TOOLS or "reranker" in ENABLED_TOOLS:
    @app.tool()
    async def reranker(query: str, documents: list[str], top_n: int = None) -> str:
        """Rerank documents based on relevance to a query using DeepInfra reranker models.
        
        Args:
            query: The search query to rank documents against
            documents: List of documents to rerank
            top_n: Optional number of top results to return. If None, returns all documents ranked.
        
        Returns:
            JSON string with ranked documents, including original index, relevance score, and text.
        """
        model = DEFAULT_MODELS["reranker"]
        try:
            # DeepInfra reranker API endpoint
            async with httpx.AsyncClient(timeout=60.0) as http_client:
                response = await http_client.post(
                    f"https://api.deepinfra.com/v1/inference/{model}",
                    headers={
                        "Authorization": f"Bearer {DEEPINFRA_API_KEY}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "query": query,
                        "documents": documents,
                        "top_n": top_n,
                        "return_documents": True
                    }
                )
                response.raise_for_status()
                result = response.json()
                
                # Format the response
                ranked_results = []
                for item in result.get("results", []):
                    # Extract document text safely
                    doc_index = item.get("index")
                    document_text = None
                    
                    # Try to get document from the item first
                    doc_obj = item.get("document")
                    if isinstance(doc_obj, dict):
                        document_text = doc_obj.get("text")
                    # Otherwise, get from original documents list using index
                    elif doc_index is not None and 0 <= doc_index < len(documents):
                        document_text = documents[doc_index]
                    
                    ranked_results.append({
                        "index": doc_index,
                        "relevance_score": item.get("relevance_score"),
                        "document": document_text
                    })
                
                return json.dumps({
                    "query": query,
                    "results": ranked_results,
                    "model": model
                }, indent=2)
        except Exception as e:
            return f"Error reranking documents: {type(e).__name__}: {str(e)}"

if "all" in ENABLED_TOOLS or "speech_recognition" in ENABLED_TOOLS:
    @app.tool()
    async def speech_recognition(audio_url: str) -> str:
        """Transcribe audio to text using DeepInfra OpenAI-compatible API (Whisper)."""
        model = DEFAULT_MODELS["speech_recognition"]
        try:
            async with httpx.AsyncClient(timeout=120.0) as http_client:
                # Download the audio file
                audio_response = await http_client.get(audio_url)
                audio_response.raise_for_status()
                audio_content = audio_response.content
            
            # Use the OpenAI-compatible Whisper API
            response = await client.audio.transcriptions.create(
                model=model,
                file=("audio.mp3", audio_content),
            )
            return response.text
        except Exception as e:
            return f"Error transcribing audio: {type(e).__name__}: {str(e)}"

if "all" in ENABLED_TOOLS or "zero_shot_image_classification" in ENABLED_TOOLS:
    @app.tool()
    async def zero_shot_image_classification(image_url: str, candidate_labels: list[str]) -> str:
        """Classify an image with zero-shot labels using DeepInfra OpenAI-compatible API (CLIP)."""
        model = DEFAULT_MODELS["zero_shot_image_classification"]
        try:
            # Use chat/completions with vision capability to get classification
            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": f"Classify this image into one of these categories: {', '.join(candidate_labels)}. Return a JSON with 'label' and 'score' fields."
                            },
                            {
                                "type": "image_url",
                                "image_url": {"url": image_url}
                            }
                        ]
                    }
                ],
                max_tokens=200,
            )
            if response.choices:
                return response.choices[0].message.content
            else:
                return "Unable to classify image"
        except Exception as e:
            return f"Error classifying image: {type(e).__name__}: {str(e)}"

if "all" in ENABLED_TOOLS or "object_detection" in ENABLED_TOOLS:
    @app.tool()
    async def object_detection(image_url: str) -> str:
        """Detect objects in an image using DeepInfra OpenAI-compatible API with multimodal model."""
        model = DEFAULT_MODELS["object_detection"]
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Analyze this image and detect all objects present. Provide a detailed list of objects you can see, their approximate locations if possible, and confidence scores. Format as JSON."
                            },
                            {
                                "type": "image_url",
                                "image_url": {"url": image_url}
                            }
                        ]
                    }
                ],
                max_tokens=500,
            )
            if response.choices:
                return response.choices[0].message.content
            else:
                return "No objects detected"
        except Exception as e:
            return f"Error detecting objects: {type(e).__name__}: {str(e)}"

if "all" in ENABLED_TOOLS or "image_classification" in ENABLED_TOOLS:
    @app.tool()
    async def image_classification(image_url: str) -> str:
        """Classify an image using DeepInfra OpenAI-compatible API with multimodal model."""
        model = DEFAULT_MODELS["image_classification"]
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Analyze this image and classify what it shows. Provide the main categories and objects visible in the image with confidence scores. Format as JSON."
                            },
                            {
                                "type": "image_url",
                                "image_url": {"url": image_url}
                            }
                        ]
                    }
                ],
                max_tokens=500,
            )
            if response.choices:
                return response.choices[0].message.content
            else:
                return "Unable to classify image"
        except Exception as e:
            return f"Error classifying image: {type(e).__name__}: {str(e)}"

if "all" in ENABLED_TOOLS or "text_classification" in ENABLED_TOOLS:
    @app.tool()
    async def text_classification(text: str) -> str:
        """Classify text using DeepInfra OpenAI-compatible API."""
        model = DEFAULT_MODELS["text_classification"]
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": f"""Analyze the following text and classify it. Determine the sentiment (positive, negative, neutral) and main category/topic. Provide your analysis in JSON format with 'sentiment' and 'category' fields.

Text: {text}

Response format: {{"sentiment": "positive/negative/neutral", "category": "topic"}}"""
                    }
                ],
                max_tokens=200,
                temperature=0.1,
            )
            if response.choices:
                return response.choices[0].message.content
            else:
                return "Unable to classify text"
        except Exception as e:
            return f"Error classifying text: {type(e).__name__}: {str(e)}"

if "all" in ENABLED_TOOLS or "token_classification" in ENABLED_TOOLS:
    @app.tool()
    async def token_classification(text: str) -> str:
        """Perform token classification (NER) using DeepInfra OpenAI-compatible API."""
        model = DEFAULT_MODELS["token_classification"]
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": f"""Perform named entity recognition on the following text. Identify all named entities (persons, organizations, locations, dates, etc.) and classify them. Provide your analysis in JSON format with an array of entities, each having 'entity', 'type', and 'position' fields.

Text: {text}

Response format: {{"entities": [{{"entity": "entity_name", "type": "PERSON/ORG/LOC/DATE/etc", "position": [start, end]}}]}}"""
                    }
                ],
                max_tokens=500,
                temperature=0.1,
            )
            if response.choices:
                return response.choices[0].message.content
            else:
                return "Unable to perform token classification"
        except Exception as e:
            return f"Error performing token classification: {type(e).__name__}: {str(e)}"

if "all" in ENABLED_TOOLS or "fill_mask" in ENABLED_TOOLS:
    @app.tool()
    async def fill_mask(text: str) -> str:
        """Fill masked tokens in text using DeepInfra OpenAI-compatible API."""
        model = DEFAULT_MODELS["fill_mask"]
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": f"""Fill in the [MASK] token in the following text with the most appropriate word. Provide the completed sentence and explain your choice.

Text: {text}

Response format: {{"filled_text": "completed sentence", "chosen_word": "word", "explanation": "reasoning"}}"""
                    }
                ],
                max_tokens=200,
                temperature=0.1,
            )
            if response.choices:
                return response.choices[0].message.content
            else:
                return "Unable to fill mask"
        except Exception as e:
            return f"Error filling mask: {type(e).__name__}: {str(e)}"

def main():
    app.run(transport='stdio')

if __name__ == "__main__":
    main()
