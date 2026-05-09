import contextvars
import logging
import os
import time
from contextlib import contextmanager
from typing import Callable, Dict, List, Optional

from openai import OpenAI

from lida.llm_types import Message, TextGenerationConfig, TextGenerationResponse, TextGenerator

logger = logging.getLogger("lida")

DEFAULT_MODEL = "deepseek-v4-pro"
DEFAULT_BASE_URL = "http://localhost:4000"


TokenSink = Callable[[str], None]
_current_sink: contextvars.ContextVar[Optional[TokenSink]] = contextvars.ContextVar(
    "lida_token_sink", default=None
)


@contextmanager
def token_sink(sink: TokenSink):
    """Install a callable that receives every streamed token chunk emitted by
    LiteLLMTextGenerator.generate() while the context is active.

    The sink is consulted via a ContextVar so it propagates across asyncio.to_thread
    boundaries but stays isolated between concurrent requests. When no sink is
    registered, generate() falls back to its non-streaming path so existing CLI
    callers are unaffected.
    """
    token = _current_sink.set(sink)
    try:
        yield
    finally:
        _current_sink.reset(token)


class LiteLLMTextGenerator(TextGenerator):
    """OpenAI-format chat completions client pointed at a LiteLLM proxy.

    Endpoint and token are read from env: LITELLM_BASE_URL, LITELLM_API_KEY.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        provider: str = "litellm",
    ):
        super().__init__(provider=provider)
        self.base_url = base_url or os.environ.get("LITELLM_BASE_URL") or DEFAULT_BASE_URL
        # LiteLLM proxies typically require a token, but allow a sentinel so the
        # OpenAI client doesn't error out when running against an open proxy.
        self.api_key = api_key or os.environ.get("LITELLM_API_KEY") or "sk-litellm-noop"
        self.model_name = model or os.environ.get("LITELLM_MODEL") or DEFAULT_MODEL
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def generate(
        self,
        messages: List[Dict[str, str]],
        config: TextGenerationConfig = TextGenerationConfig(),
        **kwargs,
    ) -> TextGenerationResponse:
        model_name = config.model or self.model_name

        request: Dict = {
            "model": model_name,
            "messages": messages,
            "n": config.n,
            "temperature": config.temperature,
        }
        if config.max_tokens:
            request["max_tokens"] = config.max_tokens
        if config.top_p is not None:
            request["top_p"] = config.top_p
        if config.stop:
            request["stop"] = config.stop

        sink = _current_sink.get()
        start = time.time()
        if sink is None:
            response = self.client.chat.completions.create(**request)
            logger.info("LiteLLM call (model=%s) completed in %.2fs", model_name, time.time() - start)

            text = [Message(role=choice.message.role or "assistant", content=choice.message.content or "") for choice in response.choices]

            usage = {}
            if response.usage:
                usage = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                }
            return TextGenerationResponse(text=text, logprobs=[], config=config, usage=usage)

        return self._generate_streaming(request, config, sink, start)

    def _generate_streaming(
        self,
        request: Dict,
        config: TextGenerationConfig,
        sink: TokenSink,
        start: float,
    ) -> TextGenerationResponse:
        request = dict(request)
        request["stream"] = True
        request["stream_options"] = {"include_usage": True}
        accumulated: List[str] = [""] * max(1, config.n)
        roles: List[str] = ["assistant"] * max(1, config.n)
        usage_obj = None

        chunks = self.client.chat.completions.create(**request)
        for chunk in chunks:
            if getattr(chunk, "usage", None):
                usage_obj = chunk.usage
            for choice in chunk.choices or []:
                idx = choice.index or 0
                while len(accumulated) <= idx:
                    accumulated.append("")
                    roles.append("assistant")
                delta = choice.delta
                if delta is None:
                    continue
                if delta.role:
                    roles[idx] = delta.role
                content = getattr(delta, "content", None)
                if not content:
                    continue
                accumulated[idx] += content
                # Only stream the first choice's tokens to the UI; n>1 callers
                # don't have a useful display for parallel streams.
                if idx == 0:
                    try:
                        sink(content)
                    except Exception:
                        logger.debug("token sink raised; ignoring", exc_info=True)

        logger.info(
            "LiteLLM streaming call (model=%s) completed in %.2fs",
            request["model"],
            time.time() - start,
        )
        text = [Message(role=roles[i], content=accumulated[i]) for i in range(len(accumulated))]
        usage = {}
        if usage_obj is not None:
            usage = {
                "prompt_tokens": getattr(usage_obj, "prompt_tokens", 0) or 0,
                "completion_tokens": getattr(usage_obj, "completion_tokens", 0) or 0,
                "total_tokens": getattr(usage_obj, "total_tokens", 0) or 0,
            }
        return TextGenerationResponse(text=text, logprobs=[], config=config, usage=usage)

    def count_tokens(self, text) -> int:
        return 0
