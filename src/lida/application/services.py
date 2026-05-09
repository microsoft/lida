from typing import Any, List, Optional
from ..domain.models import Goal, Persona, Summary, ChartExecutorResponse
from ..domain.ports import (
    ISummarizer,
    IGoalGenerator,
    IPersonaGenerator,
    IVizGenerator,
    IChartExecutor,
)


class LidaApplication:
    def __init__(
        self,
        summarizer: ISummarizer,
        goal_generator: IGoalGenerator,
        persona_generator: IPersonaGenerator,
        viz_generator: IVizGenerator,
        chart_executor: IChartExecutor,
    ):
        self.summarizer = summarizer
        self.goal_generator = goal_generator
        self.persona_generator = persona_generator
        self.viz_generator = viz_generator
        self.chart_executor = chart_executor

    def summarize(
        self,
        data: Any,
        summary_method: str = "default",
        textgen_config: Optional[Any] = None,
    ) -> Summary:
        return self.summarizer.summarize(data, summary_method, textgen_config)

    def generate_goals(
        self,
        summary: Summary,
        textgen_config: Optional[Any] = None,
        n: int = 5,
        persona: Optional[Persona] = None,
    ) -> List[Goal]:
        return self.goal_generator.generate(summary, textgen_config, n, persona)

    def generate_personas(self, summary: Summary, textgen_config: Optional[Any] = None, n: int = 5) -> List[Persona]:
        return self.persona_generator.generate(summary, textgen_config, n)

    def generate_viz(
        self,
        summary: Summary,
        goal: Goal,
        textgen_config: Optional[Any] = None,
        library: str = "altair",
    ) -> List[str]:
        return self.viz_generator.generate(summary, goal, textgen_config, library)

    def visualize(
        self,
        summary: Summary,
        goal: Goal,
        textgen_config: Optional[Any] = None,
        library: str = "altair",
        return_error: bool = False,
        data: Any = None,
    ) -> List[ChartExecutorResponse]:
        code_specs = self.generate_viz(summary, goal, textgen_config, library)
        return self.execute_chart(code_specs, data, summary, library)

    def execute_chart(
        self,
        code_specs: List[str],
        data: Any,
        summary: Summary,
        library: str = "altair",
    ) -> List[ChartExecutorResponse]:
        return self.chart_executor.execute(code_specs, data, summary, library)
