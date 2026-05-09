import json
import logging
from typing import List, Optional

from lida.llm_types import TextGenerator, TextGenerationConfig

from lida.domain.models import Goal, Persona, Summary
from lida.domain.ports import IGoalGenerator
from lida.utils import clean_code_snippet, adapt_messages_for_provider

logger = logging.getLogger("lida")

SYSTEM_INSTRUCTIONS = """
You are a an experienced data analyst who can generate a given number of insightful GOALS about data, when given a summary of the data, and a specified persona. The VISUALIZATIONS YOU RECOMMEND MUST FOLLOW VISUALIZATION BEST PRACTICES (e.g., must use bar charts instead of pie charts for comparing quantities) AND BE MEANINGFUL (e.g., plot longitude and latitude on maps where appropriate). They must also be relevant to the specified persona. Each goal must include a question, a visualization (THE VISUALIZATION MUST REFERENCE THE EXACT COLUMN FIELDS FROM THE SUMMARY), and a rationale (JUSTIFICATION FOR WHICH dataset FIELDS ARE USED and what we will learn from the visualization). Each goal MUST mention the exact fields from the dataset summary above
"""

FORMAT_INSTRUCTIONS = """
THE OUTPUT MUST BE A CODE SNIPPET OF A VALID LIST OF JSON OBJECTS. IT MUST USE THE FOLLOWING FORMAT:

```[
    { "index": 0,  "question": "What is the distribution of X", "visualization": "histogram of X", "rationale": "This tells about "} ..
    ]
```
THE OUTPUT SHOULD ONLY USE THE JSON FORMAT ABOVE.
"""


class GoalGeneratorAdapter(IGoalGenerator):
    def __init__(self, text_gen: TextGenerator):
        self.text_gen = text_gen

    def generate(
        self,
        summary: Summary,
        textgen_config: Optional[TextGenerationConfig] = None,
        n: int = 5,
        persona: Optional[Persona] = None,
    ) -> List[Goal]:
        """Generate goals given a summary of data"""

        if textgen_config is None:
            textgen_config = TextGenerationConfig(n=1)

        # Convert Summary domain object to dict-like structure for the prompt if needed,
        # or just stringify it. The original code passed `summary: dict`.
        # The prompt uses {summary} inside an f-string.
        # Ideally we should have a reliable way to serialize Summary for the prompt.
        # For now, let's cast it to dict or use its string representation if appropriate.
        # But `Summary` is a dataclass, so we can use `asdict`?
        # Or just rely on `summary.__str__` or construct a dict manually.
        # The original code passed a dict.

        summary_dict = {
            "name": summary.name,
            "file_name": summary.file_name,
            "dataset_description": summary.dataset_description,
            "fields": summary.fields,
            "field_names": summary.field_names,
        }

        user_prompt = f"""The number of GOALS to generate is {n}. The goals should be based on the data summary below, \n\n .
        {summary_dict} \n\n"""

        if not persona:
            persona = Persona(
                persona="A highly skilled data analyst who can come up with complex, insightful goals about data",
                rationale="",
            )

        user_prompt += f"""\n The generated goals SHOULD BE FOCUSED ON THE INTERESTS AND PERSPECTIVE of a '{persona.persona} persona, who is insterested in complex, insightful goals about the data. \n"""

        messages = [
            {"role": "system", "content": SYSTEM_INSTRUCTIONS},
            {
                "role": "assistant",
                "content": f"{user_prompt}\n\n {FORMAT_INSTRUCTIONS} \n\n. The generated {n} goals are: \n ",
            },
        ]

        messages = adapt_messages_for_provider(messages, self.text_gen.provider)
        result = self.text_gen.generate(messages=messages, config=textgen_config)

        try:
            json_string = clean_code_snippet(result.text[0]["content"])
            result_json = json.loads(json_string)
            if isinstance(result_json, dict):
                result_json = [result_json]
            goals = [Goal(**x) for x in result_json]
        except json.decoder.JSONDecodeError:
            error_content = result.text[0]["content"]
            logger.error(f"Error decoding JSON from LLM response: {error_content}")
            raise ValueError("The model did not return a valid JSON object while attempting generate goals. Consider using a larger model or a model with higher max token length.")
        return goals
