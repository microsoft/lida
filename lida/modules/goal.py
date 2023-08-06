import json
import logging
from lida.utils import clean_code_snippet
from llmx import TextGenerator
from lida.datamodel import Goal, TextGenerationConfig


system_prompt = """You are a an experienced data analyst as well as visualization specialist who can read dataset properties and generate insightful questions about the data.
Your output MUST be perfectly valid JSON with no preamble or explanation. Your output MUST be in THE FORM OF A VALID PYTHON LIST OF DICTIONARIES:
[{ "index": 0,  "question": "What is the distribution of X", "visualization": "histogram of X", "rationale": "This tells about "}, {"index": 1, }]
"""

logger = logging.getLogger(__name__)


class GoalExplorer():
    """Generat goals given a summary of data"""

    def __init__(self, text_gen: TextGenerator) -> None:
        self.summary = None
        self.text_gen = text_gen

    def generate(self, summary: dict, textgen_config: TextGenerationConfig, n=5) -> list[Goal]:
        """Generate goals given a summary of data"""

        user_prompt = f"""Generate {n} questions, visualization and rationale an experienced data analyst can ask given the dataset properties.
        {summary}""" + \
            """
        The visualization must mention the exact fields from the dataset. The VISUALIZATIONS YOU RECOMMEND MUST FOLLOW VISUALIZATION BEST PRACTICES (e.g., must use bar charts instead of pie charts for comparing quantities) AND BE MEANINGFUL (e.g., plot longitude and latitude on maps where appropriate). Your response must be a valid JSON  with no preamble or explanation.
        """

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "assistant", "content": user_prompt},
        ]

        result: list[Goal] = self.text_gen.generate(messages=messages, config=textgen_config)

        try:
            json_string = clean_code_snippet(result.text[0]["content"])
            result = json.loads(json_string)
            # cast each item in the list to a Goal object
            if isinstance(result, dict):
                result = [result]
            result = [Goal(**x) for x in result]
        except json.decoder.JSONDecodeError:
            logger.info(f"Error decoding JSON: {result.text[0]['content']}")
            print(f"Error decoding JSON: {result.text[0]['content']}")
            raise ValueError(
                "The model did not return a valid JSON object while attempting generate goals. Please try again.",
                result.usage,
                result.text[0]["content"])
        return result
