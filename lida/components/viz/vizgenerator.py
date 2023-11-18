from typing import Dict
from llmx import TextGenerator, TextGenerationConfig, TextGenerationResponse

from ..scaffold import ChartScaffold
from lida.datamodel import Goal


SYSTEM_PROMPT = """
You are a highly skilled data analyst and data visualization expert. Given some code template, you complete the <template> to generate a visualization or analysis code that addresses the <goal> described. The code you write MUST FOLLOW VISUALIZATION AND DATA ANALYSIS BEST PRACTICES ie. meet the specified goal, apply the right transformation, use the right visualization type, use the right data encoding, and use the right aesthetics (e.g., ensure axis are legible). The transformations you apply MUST be correct and the fields you use MUST be correct. The visualization CODE MUST BE CORRECT and MUST NOT CONTAIN ANY SYNTAX OR LOGIC ERRORS (e.g., it must consider the field types and use them correctly). 

Your output maybe code for a visualization or code for data analysis. 
"""

FORMAT_INSTRUCTIONS = """
THE OUTPUT MUST BE A CODE SNIPPET IN VALID JSON FORMAT:

```{
"visualization": "import ...",
"analysis": "import ..."
}
``` 


THE OUTPUT SHOULD ONLY USE THE JSON FORMAT ABOVE.
"""


class VizGenerator(object):
    """Generate visualizations from prompt"""

    def __init__(
        self
    ) -> None:

        self.scaffold = ChartScaffold()

    def generate(self, summary: Dict, goal: Goal,
                 textgen_config: TextGenerationConfig, text_gen: TextGenerator, library='seaborn'):
        """Generate visualization code given a summary and a goal"""

        library_template, library_instructions = self.scaffold.get_template(
            goal, library)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "system", "content": f"The dataset <summary> is : {summary} \n\n"},
            library_instructions,
            {"role": "user",
             "content":
             f"Always add a legend with various colors where appropriate. The visualization code MUST only use data fields that exist in the dataset (field_names) or fields that are transformations based on existing field_names). Only use variables that have been defined in the code or are in the dataset summary. You MUST return a FULL PYTHON PROGRAM ENCLOSED IN BACKTICKS ``` that starts with an import statement. DO NOT add any explanation. \n\n THE GENERATED CODE SOLUTION SHOULD BE CREATED BY MODIFYING THE SPECIFIED PARTS OF THE <template> BELOW \n\n {library_template} \n\n.The FINAL COMPLETED CODE BASED ON THE <template> above is ... \n\n"}]

        completions: TextGenerationResponse = text_gen.generate(
            messages=messages, config=textgen_config)
        response = [x['content'] for x in completions.text]

        return response
