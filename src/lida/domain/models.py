from typing import Any, Dict, List, Optional, Union
from pydantic.dataclasses import dataclass


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

    question: str
    visualization: str
    rationale: str
    index: Optional[int] = 0


@dataclass
class Summary:
    """A summary of a dataset"""

    name: str
    file_name: str
    dataset_description: str
    field_names: List[Any]
    fields: Optional[List[Any]] = None


@dataclass
class Persona:
    """A persona"""

    persona: str
    rationale: str


@dataclass
class ChartExecutorResponse:
    """Response from a visualization execution"""

    spec: Optional[Union[str, Dict]]  # interactive specification e.g. vegalite
    status: bool  # True if successful
    raster: Optional[str]  # base64 encoded image
    code: str  # code used to generate the visualization
    library: str  # library used to generate the visualization
    error: Optional[Dict] = None  # error message if status is False
