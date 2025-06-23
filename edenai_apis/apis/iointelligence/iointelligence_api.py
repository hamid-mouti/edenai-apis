import json
from typing import Dict, List, Optional

import requests

from edenai_apis.features import ProviderInterface, TextInterface
from edenai_apis.features.text import GenerationDataClass
from edenai_apis.utils.exception import ProviderException
from edenai_apis.loaders.data_loader import ProviderDataEnum
from edenai_apis.loaders.loaders import load_provider
from edenai_apis.utils.types import ResponseType
from edenai_apis.llmengine.llm_engine import LLMEngine
from edenai_apis.features.text.embeddings import EmbeddingsDataClass, EmbeddingDataClass

class IOIntelligenceApi(ProviderInterface, TextInterface):
    provider_name = "iointelligence"

    def __init__(self, api_keys: Dict = {}):
        self.api_settings = load_provider(
            ProviderDataEnum.KEY, self.provider_name, api_keys=api_keys
        )
        self.api_key = self.api_settings["api_key"]
        self.base_url = "https://api.intelligence.io.solutions/api/v1"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        self.llm_client = LLMEngine(
            provider_name=self.provider_name, provider_config={"api_key": self.api_key}
        )

    def text__generation(
        self,
        text: str,
        temperature: float,
        max_tokens: int,
        model: str = "mistralai/Mistral-Large-Instruct-2411",
        **kwargs
    ) -> ResponseType[GenerationDataClass]:
        url = f"{self.base_url}/chat/completions"

        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant."
            },
            {
                "role": "user",
                "content": text
            }
        ]

        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_completion_tokens": max_tokens
        }
        if max_tokens != 0:
            payload["max_tokens"] = max_tokens
        response = requests.post(url, headers=self.headers, json=payload)

        if response.status_code != 200:
            raise ProviderException(f"Error: {response.text}")

        original_response = response.json()

        return ResponseType[GenerationDataClass](
            original_response=original_response,
            standardized_response=GenerationDataClass(
                generated_text=original_response["choices"][0]["message"]["content"]
            )
        )
    
    def text__embeddings(
        self,
        texts: List[str],
        model: Optional[str] = "BAAI/bge-multilingual-gemma2",
        **kwargs,
    ) -> ResponseType[EmbeddingsDataClass]:
        url = f"{self.base_url}/embeddings"
        
        payload = {
            "input": texts,
            "model": model,
            "encoding_format": "float"
        }
        
        response = requests.post(url, headers=self.headers, json=payload)
        
        if response.status_code != 200:
            raise ProviderException(f"Error: {response.text}")
            
        original_response = response.json()
        embeddings = original_response["data"]
        sorted_embeddings = sorted(embeddings, key=lambda e: e["index"])
        #print(embeddings)
        items = [
            EmbeddingDataClass(embedding=result["embedding"])
            for result in sorted_embeddings
        ]

        standardized_response = EmbeddingsDataClass(
            items=items
        )
        
        return ResponseType[EmbeddingsDataClass](
            original_response=original_response,
            standardized_response=standardized_response
        )
    