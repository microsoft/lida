from typing import Union, List, Dict, Callable
from dataclasses import asdict
from llmx import TextGenerator
from lida.datamodel import TextGenerationConfig
from llmx import TextGenerationResponse, Message
from llmx.utils import cache_request, num_tokens_from_messages


class CustomTextGenerator(TextGenerator):
    def __init__(
        self,
        text_generation_function: Callable[[str], str],
        provider: str = "custom",
        **kwargs
    ):
        super().__init__(provider=provider, **kwargs)
        self.text_generation_function = text_generation_function

    def generate(
        self,
        messages: Union[List[Dict], str],
        config: TextGenerationConfig = TextGenerationConfig(),
        **kwargs
    ) -> TextGenerationResponse:
        use_cache = config.use_cache
        messages = self.format_messages(messages)

        if use_cache:
            response = cache_request(cache=self.cache, params={"messages": messages})
            if response:
                return TextGenerationResponse(**response)

        generation_response = self.text_generation_function(messages)
        response = TextGenerationResponse(
            text=[Message(role="system", content=generation_response)],
            logprobs=[],  # You may need to extract log probabilities from the response if needed
            usage={},
            config={},
        )

        if use_cache:
            cache_request(
                cache=self.cache, params={"messages": messages}, values=asdict(response)
            )

        return response

    def format_messages(self, messages) -> str:
        prompt = ""
        for message in messages:
            if message["role"] == "system":
                prompt += message["content"] + "\n"
            else:
                prompt += message["role"] + ": " + message["content"] + "\n"

        return prompt

    def count_tokens(self, text) -> int:
        return num_tokens_from_messages(text)