"""
Custom DeepInfra Embeddings wrapper for LangChain
This avoids loading the model locally (saves 600MB+ RAM)
"""
import os
import requests
from typing import List
from langchain_core.embeddings import Embeddings


class DeepInfraEmbeddings(Embeddings):
    """DeepInfra API-based embeddings for memory-efficient deployment"""
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-Embedding-0.6B",
        api_token: str = None,
    ):
        self.model_name = model_name
        self.api_token = api_token or os.getenv("DEEPINFRA_TOKEN")
        if not self.api_token:
            raise ValueError("DEEPINFRA_TOKEN is required")
        
        self.api_url = f"https://api.deepinfra.com/v1/inference/{model_name}"
        self.headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json",
        }
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple documents using DeepInfra API"""
        response = requests.post(
            self.api_url,
            headers=self.headers,
            json={"inputs": texts},
            timeout=int(os.getenv("DEEPINFRA_TIMEOUT_SECONDS", "8")),
        )
        response.raise_for_status()
        result = response.json()

        # Common response shape: {"embeddings": [[...], [...]]}
        if isinstance(result, dict) and "embeddings" in result and isinstance(result["embeddings"], list):
            return result["embeddings"]

        # Some models may return raw list of vectors
        if isinstance(result, list) and result and isinstance(result[0], list):
            return result

        raise ValueError(f"Unexpected DeepInfra embeddings response format: {result}")
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query using DeepInfra API"""
        return self.embed_documents([text])[0]
