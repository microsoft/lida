from typing import Dict, List, Union
from llmx import TextGenerator, TextGenerationConfig, TextGenerationResponse

from lida.modules.scaffold import ChartScaffold
from lida.datamodel import Goal, Summary

system_prompt = """
You are a helpful assistant highly skilled in revising visualization code to improve the quality of the code and visualization based on feedback.  Assume that data in plot(data) contains a valid dataframe.
You MUST return a full program. DO NOT include any preamble text. Do not include explanations or prose.
"""


class VizRepairer(object):
    """Fix visualization code based on feedback"""

    def __init__(
        self,
    ) -> None:
        self.scaffold = ChartScaffold()

    def generate(
            self, code: str, feedback: Union[str, Dict, List[Dict]],
            goal: Goal, summary: Summary, textgen_config: TextGenerationConfig,
            text_gen: TextGenerator, library='altair',):
        """Fix a code spec based on feedback"""
        library_template, library_instructions = self.scaffold.get_template(Goal(
            index=0,
            question="",
            visualization="",
            rationale=""), library)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "system", "content": f"The dataset summary is : {summary}. \n . The original goal was: {goal}."},
            {"role": "system",
             "content":
             f"You MUST use only the {library}. The resulting code MUST use the following template {library_template}. Only use variables that have been defined in the code or are in the dataset summary"},
            {"role": "user", "content": f"The existing code to be fixed is: {code}. \n Fix the code above to address the feedback: {feedback}. ONLY apply feedback that are CORRECT."}]

        # library with the following instructions {library_instructions}

        completions: TextGenerationResponse = text_gen.generate(
            messages=messages, config=textgen_config)
        return [x['content'] for x in completions.text]
