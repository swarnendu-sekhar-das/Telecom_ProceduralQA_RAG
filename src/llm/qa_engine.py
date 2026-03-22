import requests

class MockNode:
    """A dummy class to mimic LlamaIndex's source nodes for the frontend UI."""
    def __init__(self, metadata):
        self.metadata = metadata

class MockResponse:
    """A dummy class to mimic LlamaIndex's Response object for the frontend UI."""
    def __init__(self, response_text, sops):
        self.response = response_text
        self.source_nodes = [MockNode(sop) for sop in sops]
        
    def __str__(self):
        return self.response

class ProceduralQAEngine:
    """
    Acts as a bridge to the remote SecureNOC Colab API, completely 
    replacing the local LlamaIndex/Groq pipeline.
    """
    
    def __init__(self, retriever_pipeline=None):
        """
        The retriever_pipeline argument is kept so main.py doesn't break, 
        but it is ignored. All retrieval happens in Colab now.
        """
        # ⚠️ IMPORTANT: Update this with your active Ngrok URL from Colab
        self.api_url = "https://unverbosely-ascocarpous-vickey.ngrok-free.dev/ask" 
        
    def query(self, query_str: str, filters: dict = None):
        """
        1. Sends the query to the Colab API.
        2. Parses the JSON response.
        3. Wraps the result in a MockResponse to keep the UI intact.
        """
        try:
            payload = {"query": query_str}
            
            # 120s timeout in case the local Mac network is slow or Colab is thinking
            response = requests.post(self.api_url, json=payload, timeout=120)
            response.raise_for_status()
            
            data = response.json()
            answer = data.get("answer", "No answer provided by API.")
            retrieved_sops = data.get("retrieved_sops", [])
            
            # Return the mocked object so main.py parses it exactly like LlamaIndex
            return MockResponse(answer, retrieved_sops)
            
        except requests.exceptions.RequestException as e:
            return MockResponse(f"API Connection Error: {e}. Is the Colab Ngrok tunnel running?", [])