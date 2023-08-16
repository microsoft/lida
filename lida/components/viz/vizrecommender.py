import logging
import json
from lida.utils import clean_code_snippet
from ..scaffold import ChartScaffold
from llmx import TextGenerator, TextGenerationConfig, TextGenerationResponse
# from lida.modules.scaffold import ChartScaffold
from lida.datamodel import Goal, Summary


system_prompt = """
You are a helpful assistant highly skilled in recommending a DIVERSE set of visualizations as code. Your input is an example visualization code,  a summary of a dataset and an example visualization goal. Given this input, your task is to recommend an additional DIVERSE visualizations that a user may be interesting to a user. Consider different types of valid aggregations, chart types, and use different variables from the data summary. THE CODE YOU GENERATE MUST BE CORRECT AND FOLLOW VISUALIZATION BEST PRACTICES.

Your output MUST be perfect JSON in THE FORM OF A VALID JSON LIST without any additional explanation  e.g.,

[{"code": "import ...", "index":0}, .. {"code": "import ...", "index":1} ]
"""

# refactor this to return n predictions ...
logger = logging.getLogger(__name__)


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
            n=3,
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
             f"An example visualization code is: {code}. You MUST use only the {library} library with the following instructions {library_instructions}. Each recommended visualization CODE MUST use the following template {library_template}."},
            {"role": "user", "content": f"Now write code for {n} visualizations in the JSON list format. YOU MUST RETURN ONLY A JSON LIST"}
        ]

        textgen_config.messages = messages
        result: TextGenerationResponse = text_gen.generate(
            messages=messages, config=textgen_config)
        try:
            json_string = clean_code_snippet(result.text[0]["content"])
            result = json.loads(json_string)
            if isinstance(result, dict):
                result = [result]
            result = [x["code"] for x in result]
        except json.decoder.JSONDecodeError:
            logger.info(
                f"Error decoding JSON for generated visualization recommendations: {result.text[0]['content']}")
            print(
                f"Error decoding JSON for generated visualization recommendations: {result.text[0]['content']}")
            raise ValueError(
                "The model did not return a valid JSON object while attempting generate visualization recommendations. Please try again.")
        return result
