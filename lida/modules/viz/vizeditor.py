from llmx import TextGenerator, TextGenerationConfig, TextGenerationResponse
from lida.modules.scaffold import ChartScaffold
from lida.datamodel import Goal, Summary


system_prompt = """
You are a helpful assistant highly skilled in modifying visualization code based on a summary of a dataset to follow instructions. You MUST return a full program. DO NOT with NO backticks ```. DO NOT include any preamble text. Do not include explanations or prose.
"""


class VizEditor(object):
    """Generate visualizations from prompt"""

    def __init__(
        self,
        model: TextGenerator,
    ) -> None:
        self.model = model
        self.scaffold = ChartScaffold()

    def generate(
            self, code: str, summary: Summary, instructions: list[str],
            textgen_config: TextGenerationConfig, library='altair'):
        """Edit a code spec based on instructions"""

        instructions = [
            {"role": "user", "content": "modify the existing  code to " + i}
            for i in instructions]

        library_template, library_instructions = self.scaffold.get_template(Goal(
            index=0,
            question="",
            visualization="",
            rationale=""), library)
        # print("instructions", instructions)

        messages = [{"role": "system", "content": system_prompt}, {"role": "system", "content": f"The dataset summary is : {summary}"}, {"role": "system",
                                                                                                                                         "content": f"The code to be modified is: {code}.  You MUST use only the {library} library with the following instructions {library_instructions}. The resulting code MUST use the following template {library_template}"}]
        messages.extend(instructions)

        completions: TextGenerationResponse = self.model.generate(
            messages=messages, config=textgen_config)
        return [x['content'] for x in completions.text]
