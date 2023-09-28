from llmx import TextGenerator, TextGenerationConfig, TextGenerationResponse
from ..scaffold import ChartScaffold
from lida.datamodel import Goal, Summary


system_prompt = """
You are a high skilled visualization assistant that can modify a provided visualization code based on a set of instructions. Your modification should ONLY UPDATE the content of the plot(data) function/method in the code. You MUST return a full program. DO NOT include any preamble text. Do not include explanations or prose.
"""


class VizEditor(object):
    """Generate visualizations from prompt"""

    def __init__(
        self,
    ) -> None:
        self.scaffold = ChartScaffold()

    def generate(
            self, code: str, summary: Summary, instructions: list[str],
            textgen_config: TextGenerationConfig, text_gen: TextGenerator, library='altair'):
        """Edit a code spec based on instructions"""

        instruction_string = ""
        for i, instruction in enumerate(instructions):
            instruction_string += f"{i+1}. {instruction} \n"

        library_template, library_instructions = self.scaffold.get_template(Goal(
            index=0,
            question="",
            visualization="",
            rationale=""), library)
        # print("instructions", instructions)

        messages = [
            {
                "role": "system", "content": system_prompt}, {
                "role": "system", "content": f"The dataset summary is : \n\n {summary} \n\n"}, {
                "role": "system", "content": f"The code to be modified is: \n\n{code} \n\n. THE CODE YOU GENERATE MUST BE CORRECT FOLLLOW  and based on the '{library}' library and also follow these instructions instructions \n\n{library_instructions} \n\n. The resulting code MUST use the following template {library_template}"}, {
                    "role": "user", "content": f"ALL ADDITIONAL LIBRARIES USED MUST BE IMPORTED.\n YOU MUST THINK STEP BY STEP, AND CAREFULLY MODIFY THE CODE above TO MEET THE EACH OF the FOLLOWING INSTRUCTIONS: \n\n {instruction_string} \n\n. The completed modified code is below. \n"}]

        completions: TextGenerationResponse = text_gen.generate(
            messages=messages, config=textgen_config)
        return [x['content'] for x in completions.text]
