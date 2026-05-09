import json
import logging
from typing import List, Optional

from lida.llm_types import TextGenerator, TextGenerationConfig

from lida.domain.models import Persona, Summary
from lida.domain.ports import IPersonaGenerator
from lida.utils import adapt_messages_for_provider, clean_code_snippet

logger = logging.getLogger("lida")

SYSTEM_PROMPT = """You are an experienced data analyst  who can take a dataset summary and generate a list of n personas (e.g., ceo or accountant for finance related data, economist for population or gdp related data, doctors for health data, or just users) that might be critical stakeholders in exploring some data and describe rationale for why they are critical. The personas should be prioritized based on their relevance to the data. Think step by step.

Your response should be perfect JSON in the following format:
```[{"persona": "persona1", "rationale": "..."},{"persona": "persona1", "rationale": "..."}]```
"""


class PersonaGeneratorAdapter(IPersonaGenerator):
    """Generate personas given a summary of data"""

    def __init__(self, text_gen: TextGenerator) -> None:
        self.text_gen = text_gen

    def generate(
        self,
        summary: Summary,
        textgen_config: Optional[TextGenerationConfig] = None,
        n: int = 5,
    ) -> List[Persona]:
        """Generate personas given a summary of data"""

        if textgen_config is None:
            textgen_config = TextGenerationConfig(n=1)

        summary_dict = {
            "name": summary.name,
            "file_name": summary.file_name,
            "dataset_description": summary.dataset_description,
            "fields": summary.fields,
            "field_names": summary.field_names,
        }

        user_prompt = (
            f"""The number of PERSONAs to generate is {n}. Generate {n} personas in the right format given the data summary below,\n .
        {summary_dict} \n"""
            + """

        .
        """
        )

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "assistant", "content": user_prompt},
        ]

        messages = adapt_messages_for_provider(messages, self.text_gen.provider)
        result = self.text_gen.generate(messages=messages, config=textgen_config)

        try:
            json_string = clean_code_snippet(result.text[0]["content"])
            result_json = json.loads(json_string)
            if isinstance(result_json, dict):
                result_json = [result_json]
            personas = [Persona(**x) for x in result_json]
        except json.decoder.JSONDecodeError:
            logger.info(f"Error decoding JSON: {result.text[0]['content']}")
            print(f"Error decoding JSON: {result.text[0]['content']}")
            raise ValueError("The model did not return a valid JSON object while attempting generate personas.  Consider using a larger model or a model with higher max token length.")
        return personas
