# from dataclasses import dataclass
from dataclasses import field
from typing import Any, Dict, List, Optional, Union

from llmx import TextGenerationConfig
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

    def _repr_markdown_(self):
        return f"""
### Goal {self.index}
---
**Question:** {self.question}

**Visualization:** `{self.visualization}`

**Rationale:** {self.rationale}
"""


@dataclass
class Summary:
    """A summary of a dataset"""

    name: str
    file_name: str
    dataset_description: str
    field_names: List[Any]
    fields: Optional[List[Any]] = None

    def _repr_markdown_(self):
        field_lines = "\n".join([f"- **{name}:** {field}" for name,
                                field in zip(self.field_names, self.fields)])
        return f"""
## Dataset Summary

---

**Name:** {self.name}

**File Name:** {self.file_name}

**Dataset Description:**

{self.dataset_description}

**Fields:**

{field_lines}
"""


@dataclass
class Persona:
    """A persona"""
    persona: str
    rationale: str

    def _repr_markdown_(self):
        return f"""
### Persona
---

**Persona:** {self.persona}

**Rationale:** {self.rationale}
"""


@dataclass
class GoalWebRequest:
    """A Goal Web Request"""

    summary: Summary
    textgen_config: Optional[TextGenerationConfig] = field(
        default_factory=TextGenerationConfig
    )
    n: int = 5


@dataclass
class VisualizeWebRequest:
    """A Visualize Web Request"""

    summary: Summary
    goal: Goal
    library: str = "seaborn"
    textgen_config: Optional[TextGenerationConfig] = field(
        default_factory=TextGenerationConfig
    )


@dataclass
class VisualizeRecommendRequest:
    """A Visualize Recommendation Request"""

    summary: Summary
    code: str
    library: str = "seaborn"
    textgen_config: Optional[TextGenerationConfig] = field(
        default_factory=TextGenerationConfig
    )


@dataclass
class VisualizeEditWebRequest:
    """A Visualize Edit Web Request"""

    summary: Summary
    code: str
    instructions: Union[str, List[str]]
    library: str = "seaborn"
    textgen_config: Optional[TextGenerationConfig] = field(
        default_factory=TextGenerationConfig
    )


@dataclass
class VisualizeRepairWebRequest:
    """A Visualize Repair Web Request"""

    feedback: Optional[Union[str, List[str], List[Dict]]]
    code: str
    goal: Goal
    summary: Summary
    library: str = "seaborn"
    textgen_config: Optional[TextGenerationConfig] = field(
        default_factory=TextGenerationConfig
    )


@dataclass
class VisualizeExplainWebRequest:
    """A Visualize Explain Web Request"""

    code: str
    library: str = "seaborn"
    textgen_config: Optional[TextGenerationConfig] = field(
        default_factory=TextGenerationConfig
    )


@dataclass
class VisualizeEvalWebRequest:
    """A Visualize Eval Web Request"""

    code: str
    goal: Goal
    library: str = "seaborn"
    textgen_config: Optional[TextGenerationConfig] = field(
        default_factory=TextGenerationConfig
    )


@dataclass
class ChartExecutorResponse:
    """Response from a visualization execution"""

    spec: Optional[Union[str, Dict]]  # interactive specification e.g. vegalite
    status: bool  # True if successful
    raster: Optional[str]  # base64 encoded image
    code: str  # code used to generate the visualization
    library: str  # library used to generate the visualization
    error: Optional[Dict] = None  # error message if status is False

    def _repr_mimebundle_(self, include=None, exclude=None):
        bundle = {"text/plain": self.code}
        if self.raster is not None:
            bundle["image/png"] = self.raster
        if self.spec is not None:
            bundle["application/vnd.vegalite.v5+json"] = self.spec

        return bundle


@dataclass
class InfographicsRequest:
    """A request for infographics generation"""

    visualization: str
    n: int = 1
    style_prompt: Union[str, List[str]] = ""
    # return_pil: bool = False
