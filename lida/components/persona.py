import json
import logging
from lida.utils import clean_code_snippet
from llmx import TextGenerator
from lida.datamodel import Persona, TextGenerationConfig


system_prompt = """You are an experienced data analyst  who can take a dataset summary and generate a list of n personas (e.g., ceo or accountant for finance related data, economist for population or gdp related data, doctors for health data, or just users) that might be critical stakeholders in exploring some data and describe rationale for why they are critical. The personas should be prioritized based on their relevance to the data. Think step by step.

Your response should be perfect JSON in the following format:
[{"persona": "persona1", "rationale": "..."},{"persona": "persona1", "rationale": "..."}]
```
"""

logger = logging.getLogger(__name__)


class PersonaExplorer():
    """Generat personas given a summary of data"""

    def __init__(self) -> None:
        pass

    def generate(self, summary: dict, textgen_config: TextGenerationConfig,
                 text_gen: TextGenerator, n=5) -> list[Persona]:
        """Generate personas given a summary of data"""

        user_prompt = f"""The number of PERSONAs to generate is {n}. Generate {n} personas in the right format given the data summary below,\n .
        {summary} \n""" + """

        .
        """

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "assistant", "content": user_prompt},
        ]

        result = text_gen.generate(messages=messages, config=textgen_config)

        try:
            json_string = clean_code_snippet(result.text[0]["content"])
            result = json.loads(json_string)
            # cast each item in the list to a Goal object
            if isinstance(result, dict):
                result = [result]
            result = [Persona(**x) for x in result]
        except json.decoder.JSONDecodeError:
            logger.info(f"Error decoding JSON: {result.text[0]['content']}")
            print(f"Error decoding JSON: {result.text[0]['content']}")
            raise ValueError(
                "The model did not return a valid JSON object while attempting generate personas. Please try again.")
        return result
