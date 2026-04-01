import requests
import json
import time
from typing import Generator, Dict, Any, Optional
import logging

class MockNode:
    """A dummy class to mimic LlamaIndex's source nodes for the frontend UI."""
    def __init__(self, metadata: Dict[str, Any]):
        self.metadata = metadata

class MockResponse:
    """A dummy class to mimic LlamaIndex's Response object for the frontend UI."""
    def __init__(self, response_text: str, sops: list):
        self.response = response_text
        self.source_nodes = [MockNode(sop) for sop in sops]
        
    def __str__(self):
        return self.response

class StreamingResponse:
    """A class to handle streaming responses from the API."""
    def __init__(self, content: str = "", sources: list = None, complete: bool = False, error: Optional[str] = None):
        self.content = content
        self.sources = sources or []
        self.complete = complete
        self.error = error
        
    def to_dict(self) -> Dict[str, Any]:
        return {
            "content": self.content,
            "sources": self.sources,
            "complete": self.complete,
            "error": self.error
        }

class ProceduralQAEngine:
    """
    Enhanced QA Engine with streaming support for the NetRestore RAG system.
    Supports both regular and streaming queries with automatic fallback.
    """
    
    def __init__(self, retriever_pipeline=None, enable_streaming: bool = True, fallback_timeout: int = 5):
        """
        Initialize the QA Engine with streaming capabilities.
        
        Args:
            retriever_pipeline: Legacy argument (kept for compatibility)
            enable_streaming: Whether to use streaming by default
            fallback_timeout: Timeout for streaming attempts before falling back
        """
        self.api_url = "https://mia-propertied-cristopher.ngrok-free.dev/ask"
        self.stream_url = "https://mia-propertied-cristopher.ngrok-free.dev/ask-stream"
        self.enable_streaming = enable_streaming
        self.fallback_timeout = fallback_timeout
        
        # Configure logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def _parse_stream_chunk(self, chunk_data: str) -> Optional[StreamingResponse]:
        """
        Parse a single chunk from the streaming API.
        
        Args:
            chunk_data: Raw chunk data from the API
            
        Returns:
            StreamingResponse object or None if invalid chunk
        """
        try:
            if chunk_data.startswith('data: '):
                data_str = chunk_data[6:]  # Remove 'data: ' prefix
                if data_str.strip() == '[DONE]':
                    return StreamingResponse(complete=True)
                    
                chunk_json = json.loads(data_str)
                chunk_type = chunk_json.get('type')
                
                if chunk_type == 'content':
                    return StreamingResponse(content=chunk_json.get('content', ''))
                elif chunk_type == 'sources':
                    return StreamingResponse(sources=chunk_json.get('sources', []))
                elif chunk_type == 'error':
                    return StreamingResponse(error=chunk_json.get('error', 'Unknown error'))
                elif chunk_type == 'done':
                    return StreamingResponse(complete=True)
                    
        except json.JSONDecodeError as e:
            self.logger.warning(f"Failed to parse chunk JSON: {e}")
        except Exception as e:
            self.logger.error(f"Error parsing stream chunk: {e}")
            
        return None
    
    def query_stream(self, query_str: str, filters: dict = None) -> Generator[StreamingResponse, None, None]:
        """
        Streaming query that yields response chunks as they're generated.
        
        Args:
            query_str: The user's query
            filters: Optional filters (legacy parameter)
            
        Yields:
            StreamingResponse objects with content, sources, or completion status
        """
        if not self.enable_streaming:
            # Fallback to mock streaming if streaming is disabled
            yield from self.query_stream_mock(query_str, filters)
            return
            
        try:
            payload = {"query": query_str}
            
            self.logger.info(f"Starting streaming query: {query_str[:50]}...")
            
            # Use streaming with Server-Sent Events
            response = requests.post(
                self.stream_url, 
                json=payload, 
                timeout=(self.fallback_timeout, 120),  # (connect, read) timeout
                stream=True,
                headers={
                    'Accept': 'text/event-stream',
                    'Cache-Control': 'no-cache',
                    'Connection': 'keep-alive'
                }
            )
            response.raise_for_status()
            
            # Parse SSE format
            for line in response.iter_lines():
                if line:
                    line = line.decode('utf-8').strip()
                    if line:
                        parsed_chunk = self._parse_stream_chunk(line)
                        if parsed_chunk:
                            yield parsed_chunk
                            
                            # If we got an error or completion, stop streaming
                            if parsed_chunk.error or parsed_chunk.complete:
                                break
                                
        except requests.exceptions.Timeout:
            self.logger.warning("Streaming timeout, falling back to mock streaming")
            yield from self.query_stream_mock(query_str, filters)
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Streaming request failed: {e}, falling back to mock streaming")
            yield from self.query_stream_mock(query_str, filters)
        except Exception as e:
            self.logger.error(f"Unexpected error in streaming: {e}")
            yield StreamingResponse(error=f"Streaming error: {str(e)}")

    def query_stream_mock(self, query_str: str, filters: dict = None, chunk_size: int = 5, delay: float = 0.05) -> Generator[StreamingResponse, None, None]:
        """
        Enhanced mock streaming that simulates chunked responses from the regular API.
        
        Args:
            query_str: The user's query
            filters: Optional filters (legacy parameter)
            chunk_size: Number of words per chunk
            delay: Delay between chunks to simulate real streaming
            
        Yields:
            StreamingResponse objects with content chunks
        """
        try:
            # Get the full response first
            response = self.query(query_str, filters)
            full_text = response.response
            sources = []
            
            # Extract sources from the response
            if hasattr(response, 'source_nodes') and response.source_nodes:
                for node in response.source_nodes:
                    sources.append({
                        "sop_id": node.metadata.get('sop_id', 'Unknown'),
                        "vendor": node.metadata.get('vendor', 'Unknown'),
                        "severity": node.metadata.get('severity', 'UNKNOWN'),
                        "title": node.metadata.get('title', 'Untitled')
                    })
            
            # Yield sources first
            if sources:
                yield StreamingResponse(sources=sources)
            
            # Simulate streaming by yielding chunks
            words = full_text.split()
            current_chunk = ""
            
            for i, word in enumerate(words):
                current_chunk += word + " "
                
                # Yield every chunk_size words or at the end
                if (i + 1) % chunk_size == 0 or i == len(words) - 1:
                    yield StreamingResponse(content=current_chunk)
                    current_chunk = ""
                    
                    # Add delay to simulate real streaming
                    if i < len(words) - 1:
                        time.sleep(delay)
            
            # Mark as complete
            yield StreamingResponse(complete=True)
                    
        except Exception as e:
            self.logger.error(f"Error in mock streaming: {e}")
            yield StreamingResponse(error=f"Mock streaming error: {str(e)}")
    
    def query(self, query_str: str, filters: dict = None) -> MockResponse:
        """
        Regular non-streaming query with enhanced error handling.
        
        Args:
            query_str: The user's query
            filters: Optional filters (legacy parameter)
            
        Returns:
            MockResponse object with answer and sources
        """
        try:
            payload = {"query": query_str}
            
            self.logger.info(f"Sending regular query: {query_str[:50]}...")
            
            # 120s timeout in case the local Mac network is slow or Colab is thinking
            response = requests.post(self.api_url, json=payload, timeout=120)
            response.raise_for_status()
            
            data = response.json()
            answer = data.get("answer", "No answer provided by API.")
            retrieved_sops = data.get("retrieved_sops", [])
            
            self.logger.info("Regular query completed successfully")
            
            # Return the mocked object so main.py parses it exactly like LlamaIndex
            return MockResponse(answer, retrieved_sops)
            
        except requests.exceptions.Timeout:
            error_msg = "Request timeout. The Colab API may be busy or experiencing high load."
            self.logger.error(error_msg)
            return MockResponse(f"API Timeout: {error_msg}", [])
        except requests.exceptions.ConnectionError:
            error_msg = "Connection failed. Unable to reach the Colab API."
            self.logger.error(error_msg)
            return MockResponse(f"API Connection Error: {error_msg}. Is the Colab Ngrok tunnel running?", [])
        except requests.exceptions.RequestException as e:
            self.logger.error(f"API request failed: {e}")
            return MockResponse(f"API Request Error: {e}. Is the Colab Ngrok tunnel running?", [])
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse API response: {e}")
            return MockResponse(f"API Response Error: Invalid JSON response from server.", [])
        except Exception as e:
            self.logger.error(f"Unexpected error in query: {e}")
            return MockResponse(f"Unexpected Error: {e}", [])