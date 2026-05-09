"""Minimal data classes for LiteLLM-backed text generation.

These replace the corresponding types from llmx so we don't pull in its
package-level provider bootstrap (which is incompatible with our config).
The field layout is kept compatible with the bits of llmx that callers
relied on (n / temperature / max_tokens / model / stop / messages / etc.).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union


@dataclass
class Message:
    role: str
    content: str

    def __getitem__(self, key: str):
        return getattr(self, key)


@dataclass
class TextGenerationConfig:
    n: int = 1
    temperature: float = 0.1
    max_tokens: Optional[int] = None
    top_p: float = 1.0
    top_k: int = 50
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    provider: Optional[str] = None
    model: Optional[str] = None
    stop: Union[List[str], str, None] = None
    use_cache: bool = True
    messages: Optional[List[Dict[str, str]]] = None


@dataclass
class TextGenerationResponse:
    text: List[Message]
    config: Any
    logprobs: Optional[Any] = None
    usage: Optional[Any] = field(default_factory=dict)
    response: Optional[Any] = None


class TextGenerator(ABC):
    """Abstract base. Concrete implementations talk to a backend (LiteLLM, etc.)."""

    def __init__(self, provider: str = "litellm", model_name: Optional[str] = None):
        self.provider = provider
        self.model_name = model_name or ""

    @abstractmethod
    def generate(
        self,
        messages: Union[List[Dict], str],
        config: TextGenerationConfig = TextGenerationConfig(),
        **kwargs,
    ) -> TextGenerationResponse: ...

    @abstractmethod
    def count_tokens(self, text) -> int: ...
