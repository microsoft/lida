
from llmx import TextGenerator, TextGenerationConfig, TextGenerationResponse

from lida.datamodel import Goal

system_prompt = """
You are a helpful assistant highly skilled in evaluating the quality of a given visualization code by providing a score from 1 (bad) - 10 (good) while providing clear rationale. YOU MUST CONSIDER VISUALIZATION BEST PRACTICES for each evaluation. Specifically, you can carefully evaluate the code across the following dimensions
- bugs (bugs):  are there bugs, logic errors, syntax error or typos? Are there any reasons why the code may fail to compile? How should it be fixed? If ANY bug exists, the bug score MUST be less than 5.
- Data transformation (transformation): Is the data transformed appropriately for the visualization type? E.g., is the dataset appropriated filtered, aggregated, or grouped  if needed?
- Goal compliance (compliance): how well the code meets the specified visualization goals?
- Visualization type (type): CONSIDERING BEST PRACTICES, is the visualization type appropriate for the data and intent? Is there a visualization type that would be more effective in conveying insights? If a different visualization type is more appropriate, the score MUST be less than 5.
- Data encoding (encoding): Is the data encoded appropriately for the visualization type?
- aesthetics (aesthetics): Are the aesthetics of the visualization appropriate for the visualization type and the data?

You must provide a score for each of the above dimensions.  Assume that data in chart = plot(data) contains a valid dataframe for the dataset. The `plot` function returns a chart (e.g., matplotlib, seaborn etc object).

Your output MUST be perfect JSON in THE FORM OF A VALID PYTHON LIST OF DICTIONARIES e.g.,
[ { "dimension":  "bugs",  "score": 1, "rationale": " .."}, { "dimension":  "type",  "score": 1, "rationale": " .."},  ..]
"""


class VizEvaluator(object):
    """Generate visualizations Explanations given some code"""

    def __init__(
        self,
        model: TextGenerator,
    ) -> None:
        self.model = model

    def generate(self, code: str, goal: Goal,
                 textgen_config: TextGenerationConfig, library='altair'):
        """Generate a visualization explanation given some code"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "assistant",
             "content": f"Generate an evaluation given the goal and code below in {library}. The goal is {goal.question} and the code is {code}.\n=======\n. Think step by step and provide an evaluation."},
        ]

        # print(messages)

        completions: TextGenerationResponse = self.model.generate(
            messages=messages, config=textgen_config)
        return [x['content'] for x in completions.text]
