# Visualization manager class that handles the visualization of the data with the following methods

# summarize data given a df
# generate goals given a summary
# generate generate visualization specifications given a summary and a goal
# execute the specification given some data

import os
from typing import List, Union
import logging

import pandas as pd
from llmx import llm, TextGenerator
from lida.modules import VizEditor, VizExplainer, VizRepairer
from lida.datamodel import Goal, Summary, TextGenerationConfig
from lida.utils import read_dataframe
from lida.modules import Summarizer, GoalExplorer, VizGenerator, ChartExecutor, VizEvaluator

import lida.web as lida


logger = logging.getLogger(__name__)


class Manager(object):
    def __init__(self, text_gen: TextGenerator = None) -> None:
        self.text_gen = text_gen or llm()
        self.summarizer = Summarizer()
        self.goal = GoalExplorer()
        self.vizgen = VizGenerator()
        self.vizeditor = VizEditor()
        self.executor = ChartExecutor()
        self.explainer = VizExplainer()
        self.evaluator = VizEvaluator()
        self.repairer = VizRepairer()
        self.data = None

    def check_textgen(self, config: TextGenerationConfig):
        """Check if self.text_gen is the same as the config passed in. If not, update self.text_gen"""

        if self.text_gen.provider != config.provider:
            logger.info(
                f"Switchging Text Generator Provider from {self.text_gen.provider} to {config.provider}")
            self.text_gen = llm(provider=config.provider)

    def summarize(
        self,
        data: Union[pd.DataFrame, str],
        file_name="",
        n_samples: int = 3,
        enrich: bool = False,
        textgen_config: TextGenerationConfig = TextGenerationConfig(n=1, temperature=0),
    ):

        self.check_textgen(config=textgen_config)

        if isinstance(data, str):
            file_name = data.split("/")[-1]
            data = read_dataframe(data)

        self.data = data
        # self.data = data
        return self.summarizer.summarize(
            data=self.data, text_gen=self.text_gen, file_name=file_name, n_samples=n_samples,
            enrich=enrich, textgen_config=textgen_config)

    def generate_goals(
            self, summary, textgen_config: TextGenerationConfig = TextGenerationConfig(),
            n=5):
        self.check_textgen(config=textgen_config)

        return self.goal.generate(summary=summary, text_gen=self.text_gen,
                                  textgen_config=textgen_config, n=n)

    def generate_viz(
        self,
        summary,
        goal,
        textgen_config: TextGenerationConfig = TextGenerationConfig(),
        library="seaborn",
    ):

        print("Generating viz ....")
        self.check_textgen(config=textgen_config)
        return self.vizgen.generate(
            summary=summary, goal=goal, textgen_config=textgen_config, text_gen=self.text_gen,
            library=library)

    def execute_viz(
        self,
        code_specs,
        data,
        summary: Summary,
        library: str = "seaborn",
        return_error: bool = False,
    ):

        if data is None:
            root_file_path = os.path.dirname(os.path.abspath(lida.__file__))
            print(root_file_path)
            data = read_dataframe(
                os.path.join(root_file_path, "files/data", summary.file_name)
            )

        # col_properties = summary.properties

        return self.executor.execute(
            code_specs=code_specs,
            data=data,
            summary=summary,
            library=library,
            return_error=return_error,
        )

    def edit_viz(
        self,
        code,
        summary: Summary,
        instructions: List[str],
        textgen_config: TextGenerationConfig = TextGenerationConfig(),
        library: str = "seaborn",
    ):
        """Edit a visualization code given a set of instructions

        Args:
            code (_type_): _description_
            instructions (List[Dict]): A list of instructions

        Returns:
            _type_: _description_
        """

        self.check_textgen(config=textgen_config)

        if isinstance(instructions, str):
            instructions = [instructions]

        return self.vizeditor.generate(
            code=code,
            summary=summary,
            instructions=instructions,
            textgen_config=textgen_config,
            text_gen=self.text_gen,
            library=library,
        )

    def repair_viz(
        self,
        code,
        goal: Goal,
        summary: Summary,
        feedback,
        textgen_config: TextGenerationConfig = TextGenerationConfig(),
        library: str = "seaborn",
    ):
        """ Repair a visulization given some feedback"""
        self.check_textgen(config=textgen_config)
        return self.repairer.generate(
            code=code,
            feedback=feedback,
            goal=goal,
            summary=summary,
            textgen_config=textgen_config,
            text_gen=self.text_gen,
            library=library,
        )

    def explain_viz(
        self,
        code,
        textgen_config: TextGenerationConfig = TextGenerationConfig(),
        library: str = "seaborn",
    ):
        """Explain a visualization code given a set of instructions

        Args:
            code (_type_): _description_
            instructions (List[Dict]): A list of instructions

        Returns:
            _type_: _description_
        """
        self.check_textgen(config=textgen_config)
        return self.explainer.generate(
            code=code,
            textgen_config=textgen_config,
            text_gen=self.text_gen,
            library=library,
        )

    def evaluate_viz(
        self,
        code,
        goal: Goal,
        textgen_config: TextGenerationConfig = TextGenerationConfig(),
        library: str = "seaborn",
    ):
        """Evaluate a visualization code given a goal

        Args:
            code (_type_): _description_
            goal (Goal): A visualization goal

        Returns:
            _type_: _description_
        """

        self.check_textgen(config=textgen_config)

        return self.evaluator.generate(
            code=code,
            goal=goal,
            textgen_config=textgen_config,
            text_gen=self.text_gen,
            library=library,
        )
