
import json
from ...datamodel import Persona
from ...utils import clean_code_snippet
from llmx import TextGenerator, TextGenerationConfig, TextGenerationResponse
from ..scaffold import ChartScaffold


SYSTEM_PROMPT = """
You are a  highly skilled data visualization and accessiblity expert who can provide detailed explanations of the plot(data: pd.DataFrame) method in the provided code. You divide the code into sections (accessiblity, transofrmation and visualization) and provide a description of each section and an explanation. NOTE: The accessbility section should be detailed enough to be an alt-text description of the chart image.
You can explain code across the following 3 sections:
1. accessibility: This should be a description of the appearance and SHOULD MENTION components of the chart (colors, chart type, axis names, title, language etc), the goal of the chart etc. 
2. transformation: This should describe the section of the code that applies any kind of data transformation (filtering, aggregation, grouping, null value handling etc)
3. visualization: step by step description of the code that creates or modifies the presented visualization.

"""

FORMAT_PROMPT = """
Your output MUST be perfect JSON in THE FORM OF A VALID LIST of JSON OBJECTS WITH PROPERLY ESCAPED SPECIAL CHARACTERS e.g.,

```[
    {"section": "accessibility", "code": "None", "explanation": "The plot is a .."}  , {"section": "transformation", "code": "..", "explanation": ".."}  ,  {"section": "visualization", "code": "..", "explanation": ".."}
    ]```

The code part of the dictionary must come from the provided code and should cover the explanation. The explanation part of the dictionary must be a string. The section part of the dictionary must be one of "accessibility", "transformation", "visualization" with no repetition. THE LIST MUST HAVE EXACTLY 3 JSON OBJECTS [{}, {}, {}].  THE GENERATED JSON  MUST BE A LIST IE START AND END WITH A SQUARE BRACKET.
"""


class VizExplainer(object):
    """Generate visualizations Explanations given some code"""

    def __init__(
        self,
    ) -> None:
        self.scaffold = ChartScaffold()

    def generate(
            self, code: str,
            textgen_config: TextGenerationConfig, text_gen: TextGenerator,  persona: Persona = None):
        """Generate a visualization explanation given some code"""

        persona_prompt = ""
        if persona:
            persona_prompt = f""" The explanation should be focused on the interests and perspective of a '{persona.persona}' persona. \n"""

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT + persona_prompt},
            {"role": "assistant",
                "content": f"The provided code to be explained is {code}.\n=======\n"},
            {"role": "user",
             "content": f"{FORMAT_PROMPT}. \n\n. The structured explanation for the code above is \n\n"}
        ]

        completions: TextGenerationResponse = text_gen.generate(
            messages=messages, config=textgen_config)

        completions = [clean_code_snippet(x['content'])
                       for x in completions.text]
        explanations = []

        for completion in completions:
            try:
                exp = json.loads(completion)
                explanations.append(exp)
            except Exception as e:
                print("Error parsing completion", completion, str(e))
        return explanations
