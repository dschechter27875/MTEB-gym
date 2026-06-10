"""
LLM client adapters.

All clients expose: .chat(messages: list[dict]) -> str

Usage:
    from gym.clients import AnthropicClient, OpenAIClient, LiteLLMClient

    client = AnthropicClient(model="claude-opus-4-6")
    response = client.chat([{"role": "user", "content": "hello"}])
"""
from __future__ import annotations

from typing import Any, Optional


class AnthropicClient:
    """Adapter for Anthropic Claude models."""

    def __init__(
        self,
        model: str = "claude-sonnet-4-6",
        max_tokens: int = 512,
        api_key: Optional[str] = None,
    ):
        import anthropic
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.max_tokens = max_tokens

    def chat(self, messages: list[dict]) -> str:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=messages,
        )
        return response.content[0].text


class OpenAIClient:
    """Adapter for OpenAI models (also works for Azure, Together, etc.)."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        max_tokens: int = 512,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.max_tokens = max_tokens

    def chat(self, messages: list[dict]) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=self.max_tokens,
        )
        return response.choices[0].message.content


class GeminiClient:
    """Adapter for Google Gemini models (natural reference model choice)."""

    def __init__(
        self,
        model: str = "gemini-1.5-flash",
        api_key: Optional[str] = None,
    ):
        import google.generativeai as genai
        if api_key:
            genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model)

    def chat(self, messages: list[dict]) -> str:
        # Convert OpenAI-style messages to Gemini format
        parts = [m["content"] for m in messages if m["role"] == "user"]
        response = self.model.generate_content(" ".join(parts))
        return response.text


class LiteLLMClient:
    """
    Universal adapter via LiteLLM — supports 100+ providers.
    pip install litellm
    """

    def __init__(self, model: str = "gpt-4o-mini", max_tokens: int = 512, **kwargs):
        self.model = model
        self.max_tokens = max_tokens
        self.kwargs = kwargs

    def chat(self, messages: list[dict]) -> str:
        from litellm import completion
        response = completion(
            model=self.model,
            messages=messages,
            max_tokens=self.max_tokens,
            **self.kwargs,
        )
        return response.choices[0].message.content


class MockLLMClient:
    """
    Deterministic mock for testing — no API calls.
    Alternates between picking A and B based on query hash.
    """

    def chat(self, messages: list[dict]) -> str:
        import hashlib
        content = messages[-1]["content"]
        h = int(hashlib.md5(content[:50].encode()).hexdigest(), 16)
        winner = "A" if h % 3 != 0 else ("B" if h % 3 == 1 else "tie")
        return f'{{"winner": "{winner}", "confidence": "medium", "reasoning": "mock judgment"}}'


class Qwen3Client:
    """
    Adapter for Qwen3-4B-Instruct via HuggingFace Inference API.
    Free to use — no API key needed for public models.
    """

    def __init__(
        self,
        model: str = "Qwen/Qwen3-4B-Instruct",
        max_tokens: int = 512,
        token: str = None,
    ):
        from huggingface_hub import InferenceClient
        self.client = InferenceClient(model=model, token=token)
        self.model = model
        self.max_tokens = max_tokens

    def chat(self, messages: list[dict]) -> str:
        response = self.client.chat_completion(
            messages=messages,
            max_tokens=self.max_tokens,
        )
        return response.choices[0].message.content
