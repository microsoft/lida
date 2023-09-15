import logging
import json
from lida.utils import clean_code_snippet
from ..scaffold import ChartScaffold
from llmx import TextGenerator, TextGenerationConfig, TextGenerationResponse
# from lida.modules.scaffold import ChartScaffold
from lida.datamodel import Goal, Summary


system_prompt = """
You are a helpful assistant highly skilled in recommending a DIVERSE set of visualization code. Your input is an example visualization code,  a summary of a dataset and an example visualization goal. Given this input, your task is to recommend an additional DIVERSE visualizations that a user may be interested. Your output considers different types of valid aggregations, chart types, clearer ways of displaying information and uses different variables from the data summary. THE CODE YOU GENERATE MUST BE CORRECT (follow the language syntax and syntax of the visualization grammar) AND FOLLOW VISUALIZATION BEST PRACTICES.

Your output MUST be a n code snippets separated by ******* (5 asterisks). For example

```python
# code snippet 1
import  ...
....
```
*****

```python
# code snippet 2
import ...
....
```

```python
# code snippet n
import ...
....
```

Do not include any text or explanation or prose. EACH CODE SNIPPET MUST BE A FULL PROGRAM (IT MUST IMPORT ALL THE LIBRARIES THAT ARE USED AND MUST CONTAIN plot(data) method) THAT FOLLOWS THE STRUCTURE OF THE EXAMPLE VISUALIZATION CODE.
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
            library='seaborn'):
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
             f"An example visualization code is: ```{code}```. You MUST use only the {library} library. \n Each of your recommended code MUST ONLY MODIFY the content of the plot(data) method in the code above. \n"},
            {"role": "user", "content": f"Recommend {n} (n=({n})) visualizations in the format specified. \n."}]

        textgen_config.messages = messages
        result: TextGenerationResponse = text_gen.generate(
            messages=messages, config=textgen_config)
        try:
            snippets = result.text[0]["content"].split("*****")
            result = [clean_code_snippet(x) for x in snippets]
        except json.decoder.JSONDecodeError:
            logger.info(
                f"Error decoding JSON for generated visualization recommendations: {result.text[0]['content']}")
            print(
                f"Error decoding JSON for generated visualization recommendations: {result.text[0]['content']}")
            raise ValueError(
                "The model did not return a valid JSON object while attempting generate visualization recommendations. Please try again.")
        return result
