from typing import Any, List, Optional, Protocol, runtime_checkable
from .models import Goal, Persona, Summary, ChartExecutorResponse


@runtime_checkable
class ISummarizer(Protocol):
    def summarize(
        self,
        data: Any,
        summary_method: str = "default",
        textgen_config: Optional[Any] = None,
    ) -> Summary: ...


@runtime_checkable
class IGoalGenerator(Protocol):
    def generate(
        self,
        summary: Summary,
        textgen_config: Optional[Any] = None,
        n: int = 5,
        persona: Optional[Persona] = None,
    ) -> List[Goal]: ...


@runtime_checkable
class IPersonaGenerator(Protocol):
    def generate(self, summary: Summary, textgen_config: Optional[Any] = None, n: int = 5) -> List[Persona]: ...


@runtime_checkable
class IVizGenerator(Protocol):
    def generate(
        self,
        summary: Summary,
        goal: Goal,
        textgen_config: Optional[Any] = None,
        library: str = "altair",
    ) -> List[str]:
        """Returns a list of code snippets"""
        ...


@runtime_checkable
class IChartExecutor(Protocol):
    def execute(
        self,
        code_specs: List[str],
        data: Any,
        summary: Summary,
        library: str = "altair",
    ) -> List[ChartExecutorResponse]: ...
