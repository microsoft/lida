from llmx import TextGenerator, TextGenerationConfig, TextGenerationResponse
from ..scaffold import ChartScaffold
from lida.datamodel import Goal, Summary


system_prompt = """
You are a high skilled visualization assistant that can modify a provided visualization code based on a set of instructions. You MUST return a full program. DO NOT include any preamble text. Do not include explanations or prose.
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
                "role": "system", "content": f"The modifications you make MUST BE CORRECT and  based on the '{library}' library and also follow these instructions \n\n{library_instructions} \n\n. The resulting code MUST use the following template \n\n {library_template} \n\n "}, {
                    "role": "user", "content": f"ALL ADDITIONAL LIBRARIES USED MUST BE IMPORTED.\n The code to be modified is: \n\n{code} \n\n. YOU MUST THINK STEP BY STEP, AND CAREFULLY MODIFY ONLY the content of the plot(..) method TO MEET EACH OF THE FOLLOWING INSTRUCTIONS: \n\n {instruction_string} \n\n. The completed modified code THAT FOLLOWS THE TEMPLATE above is. \n"}]

        completions: TextGenerationResponse = text_gen.generate(
            messages=messages, config=textgen_config)
        return [x['content'] for x in completions.text]
