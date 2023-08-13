
from llmx import TextGenerator, TextGenerationConfig, TextGenerationResponse
from lida.modules.scaffold import ChartScaffold
from lida.datamodel import Goal, Summary


system_prompt = """
You are a helpful assistant highly skilled in recommending a DIVERSE set of visualizations. Your input is an existing visualization, and  a summary of a dataset and an example visualization goal. Given this input, your task is to recommend an additional DIVERSE visualization that a user may be interesting to a user. Consider different types of valid aggregations, chart types, and use different variables from the data summary. THE CODE YOU GENERATE MUST BE CORRECT AND FOLLOW VISUALIZATION BEST PRACTICES. You MUST return a full program.  DO NOT include any preamble text. Do not include explanations or prose.
"""


class VizRecommender(object):
    """Generate visualizations from prompt"""

    def __init__(
        self,
    ) -> None:
        self.scaffold = ChartScaffold()

    def generate(
            self, code: str, summary: Summary,
            textgen_config: TextGenerationConfig,
            text_gen: TextGenerator,
            library='altair'):
        """Recommend a code spec based on existing visualization"""

        library_template, library_instructions = self.scaffold.get_template(Goal(
            index=0,
            question="",
            visualization="",
            rationale=""), library)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "system", "content": f"The dataset summary is : {summary}"},
            {"role": "system",
             "content":
             f"The original visualization code is: {code}.  You MUST use only the {library} library with the following instructions {library_instructions}. The resulting code MUST use the following template {library_template}. Now write code for an additional visualization that a user may be interested in given the goal and the dataset summary above."}]

        textgen_config.messages = messages
        completions: TextGenerationResponse = text_gen.generate(
            messages=messages, config=textgen_config)
        return [x['content'] for x in completions.text]
