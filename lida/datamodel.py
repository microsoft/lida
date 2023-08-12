# from dataclasses import dataclass
from dataclasses import field
from typing import Any, Dict, List, Optional, Union
from pydantic.dataclasses import dataclass
from llmx import TextGenerationConfig


@dataclass
class VizGeneratorConfig:
    """Configuration for a visualization generation"""
    hypothesis: str
    data_summary: Optional[str] = ""
    data_filename: Optional[str] = "cars.csv"


@dataclass
class CompletionResult:
    text: str
    logprobs: Optional[List[float]]
    prompt: str
    suffix: str


@dataclass
class UploadUrl:
    """Response from a text generation"""
    url: str


@dataclass
class Goal:
    """A visualization goal"""
    index: int
    question: str
    visualization: str
    rationale: str


@dataclass
class Summary:
    """A summary of a dataset"""
    name: str
    file_name: str
    dataset_description: str
    field_names: List[Any]
    fields: List[Any]


@dataclass
class GoalWebRequest:
    """A Goal Web Request"""
    summary: Summary
    textgen_config: Optional[TextGenerationConfig] = field(default_factory=TextGenerationConfig)
    n: int = 5


@dataclass
class VisualizeWebRequest:
    """A Visualize Web Request"""
    summary: Summary
    goal: Goal
    library: str = "seaborn"
    textgen_config: Optional[TextGenerationConfig] = field(default_factory=TextGenerationConfig)


@dataclass
class VisualizeEditWebRequest:
    """A Visualize Edit Web Request"""
    summary: Summary
    code: str
    instructions: Union[str, List[str]]
    library: str = "seaborn"
    textgen_config: Optional[TextGenerationConfig] = field(default_factory=TextGenerationConfig)


@dataclass
class VisualizeRepairWebRequest:
    """A Visualize Repair Web Request"""
    feedback: Optional[Union[str, List[str], List[Dict]]]
    code: str
    goal: Goal
    summary: Summary
    library: str = "seaborn"
    textgen_config: Optional[TextGenerationConfig] = field(default_factory=TextGenerationConfig)


@dataclass
class VisualizeExplainWebRequest:
    """A Visualize Explain Web Request"""
    code: str
    library: str = "seaborn"
    textgen_config: Optional[TextGenerationConfig] = field(default_factory=TextGenerationConfig)


@dataclass
class VisualizeEvalWebRequest:
    """A Visualize Eval Web Request"""
    code: str
    goal: Goal
    library: str = "seaborn"
    textgen_config: Optional[TextGenerationConfig] = field(default_factory=TextGenerationConfig)


@dataclass
class ChartExecutorResponse:
    """Response from a visualization execution"""
    spec: Optional[Union[str, Dict]]  # interactive specification e.g. vegalite
    status: bool         # True if successful
    raster: Optional[str]  # base64 encoded image
    code: str           # code used to generate the visualization
    library: str     # library used to generate the visualization
    error: Optional[Dict] = None       # error message if status is False
