
import json
from lida.utils import clean_code_snippet
from llmx import TextGenerator, TextGenerationConfig, TextGenerationResponse
from lida.modules.scaffold import ChartScaffold


system_prompt = """
You are a helpful assistant highly skilled in providing helpful, structured explanations of visualization of the plot(data: pd.DataFrame) method in the provided code. You divide the code into sections and provide a description of each section and a rationale. The first section should be named "accessibility" and describe the physical appearance of the chart (colors, chart type etc), the goal of the chart, as well the main insights from the chart.
You can explain code across several dimensions:
1. accessibility: the physical appearance of the chart (colors, chart type etc), the goal of the chart, as well the main insights from the chart.
2. transformation: This should describe the section of the code that applies any kind of data transformation (filtering, aggregation, grouping, null value handling etc)
3. visualization: This should describe the section of the code that creates the visualization (e.g., plot(data)), describing the type of visualization, the aesthetics e.g. how the axis are modified etc, the data encoding.

Your output MUST be perfect JSON in THE FORM OF A VALID PYTHON LIST OF DICTIONARIES e.g.,
[{"section": "accessibility", "code": "None", "explanation": ".."},  { "section":  "transformation",  "code": "..", "explanation": ".."}, { "section":  "visualization",  "code": "..", "explanation": ".."}]
"""


class VizExplainer(object):
    """Generate visualizations Explanations given some code"""

    def __init__(
        self,
        model: TextGenerator,
    ) -> None:
        self.model = model
        self.scaffold = ChartScaffold()

    def generate(
            self, code: str,
            textgen_config: TextGenerationConfig, library='seaborn'):
        """Generate a visualization explanation given some code"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"The code to be explained is {code}.\n=======\n"},
            {"role": "assistant",
             "content": f"Generate a structured explanation of the code explanation for the code above."}
        ]

        completions: TextGenerationResponse = self.model.generate(
            messages=messages, config=textgen_config)

        completions = [clean_code_snippet(x['content']) for x in completions.text]
        explanations = []
        for completion in completions:
            try:
                exp = json.loads(completion)
                explanations.append(exp)
            except Exception as e:
                print("Error parsing completion", completion, str(e))
        return explanations
