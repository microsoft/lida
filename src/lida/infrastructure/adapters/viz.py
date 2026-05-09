from typing import List, Optional

from lida.llm_types import TextGenerator, TextGenerationConfig, TextGenerationResponse

from lida.components.scaffold import ChartScaffold
from lida.domain.models import Goal, Summary
from lida.domain.ports import IVizGenerator
from lida.utils import adapt_messages_for_provider


SYSTEM_PROMPT = """
You are a helpful assistant highly skilled in writing PERFECT code for visualizations. Given some code template, you complete the template to generate a visualization given the dataset and the goal described. The code you write MUST FOLLOW VISUALIZATION BEST PRACTICES ie. meet the specified goal, apply the right transformation, use the right visualization type, use the right data encoding, and use the right aesthetics (e.g., ensure axis are legible). The transformations you apply MUST be correct and the fields you use MUST be correct. The visualization CODE MUST BE CORRECT and MUST NOT CONTAIN ANY SYNTAX OR LOGIC ERRORS (e.g., it must consider the field types and use them correctly). You MUST first generate a brief plan for how you would solve the task e.g. what transformations you would apply e.g. if you need to construct a new column, what fields you would use, what visualization type you would use, what aesthetics you would use, etc. .
"""


class VizGeneratorAdapter(IVizGenerator):
    """Generate visualizations from prompt"""

    def __init__(self, text_gen: TextGenerator) -> None:
        self.text_gen = text_gen
        self.scaffold = ChartScaffold()

    def generate(
        self,
        summary: Summary,
        goal: Goal,
        textgen_config: Optional[TextGenerationConfig] = None,
        library: str = "altair",
    ) -> List[str]:
        """Generate visualization code given a summary and a goal"""

        if textgen_config is None:
            textgen_config = TextGenerationConfig(n=1)

        # Iterate
        # The scaffold expects a Goal object, which we have.

        # summary needs to be passed to prompt.
        # Original code used `summary: Dict`.
        summary_dict = {
            "name": summary.name,
            "file_name": summary.file_name,
            "dataset_description": summary.dataset_description,
            "fields": summary.fields,
            "field_names": summary.field_names,
        }

        library_template, library_instructions = self.scaffold.get_template(goal, library)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "system",
                "content": f"The dataset summary is : {summary_dict} \n\n",
            },
            library_instructions,
            {
                "role": "user",
                "content": f"Always add a legend with various colors where appropriate. The visualization code MUST only use data fields that exist in the dataset (field_names) or fields that are transformations based on existing field_names). Only use variables that have been defined in the code or are in the dataset summary. You MUST return a FULL PYTHON PROGRAM ENCLOSED IN BACKTICKS ``` that starts with an import statement. DO NOT add any explanation. \n\n THE GENERATED CODE SOLUTION SHOULD BE CREATED BY MODIFYING THE SPECIFIED PARTS OF THE TEMPLATE BELOW \n\n {library_template} \n\n.The FINAL COMPLETED CODE BASED ON THE TEMPLATE above is ... \n\n",
            },
        ]

        messages = adapt_messages_for_provider(messages, self.text_gen.provider)

        completions: TextGenerationResponse = self.text_gen.generate(messages=messages, config=textgen_config)
        response = [x["content"] for x in completions.text]

        return response
