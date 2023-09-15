import json
import logging
from lida.utils import clean_code_snippet
from llmx import TextGenerator
from lida.datamodel import Goal, TextGenerationConfig, Persona


system_prompt = """You are a an experienced data analyst who can generate a given number of insightful GOALS about data, when given a summary of the data, and a specified persona. The VISUALIZATIONS YOU RECOMMEND MUST FOLLOW VISUALIZATION BEST PRACTICES (e.g., must use bar charts instead of pie charts for comparing quantities) AND BE MEANINGFUL (e.g., plot longitude and latitude on maps where appropriate). They must also be relevant to the specified persona. The goal must include a question, a visualization, and a rationale (justification of what we will learn from the visualization).

The GOALS that you recommend must mention the exact fields from the dataset summary above. Your OUTPUT MUST BE ONLY A CODE SNIPPET of a JSON LIST in the format:
```[{ "index": 0,  "question": "What is the distribution of X", "visualization": "histogram of X", "rationale": "This tells about "}, .. ]
```
"""

logger = logging.getLogger(__name__)


class GoalExplorer():
    """Generat goals given a summary of data"""

    def __init__(self) -> None:
        pass

    def generate(self, summary: dict, textgen_config: TextGenerationConfig,
                 text_gen: TextGenerator, n=5, persona: Persona = None) -> list[Goal]:
        """Generate goals given a summary of data"""

        user_prompt = f"""The number of GOALS to generate is {n}. Generate {n} GOALS in the right format given the data summary below,\n .
        {summary} \n"""

        if persona:
            user_prompt += f"""\n The generated goals should be focused on the interests of a  {persona.persona} persona. \n"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "assistant", "content": user_prompt},
        ]

        result: list[Goal] = text_gen.generate(messages=messages, config=textgen_config)

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
                "The model did not return a valid JSON object while attempting generate goals. Please try again.")
        return result
