import sys
from unittest.mock import MagicMock
import logging

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("lida")

# Mock dependencies BEFORE importing lida
sys.modules["llmx"] = MagicMock()
sys.modules["diskcache"] = MagicMock()
sys.modules["tiktoken"] = MagicMock()
sys.modules["pydantic"] = MagicMock()  # Real pydantic might be needed?
# Actually we need real Pydantic for models.
del sys.modules["pydantic"]

# Force matching backend or mock
sys.modules["matplotlib"] = MagicMock()
sys.modules["matplotlib.pyplot"] = MagicMock()
sys.modules["seaborn"] = MagicMock()
sys.modules["altair"] = MagicMock()
sys.modules["plotly"] = MagicMock()
sys.modules["plotly.io"] = MagicMock()
sys.modules["plotly.graph_objects"] = MagicMock()

# Mock plt.subplots to return tuple
mock_plt = MagicMock()
mock_fig = MagicMock()
mock_ax = MagicMock()
mock_plt.subplots.return_value = (mock_fig, mock_ax)
mock_plt.figure.return_value = mock_fig
mock_fig.add_subplot.return_value = mock_ax


# Mock savefig to write to buffer explicitly
def savefig_side_effect(*args, **kwargs):
    print("savefig called!")
    if args and hasattr(args[0], "write"):
        args[0].write(b"fake_raster_data")


mock_plt.savefig.side_effect = savefig_side_effect
sys.modules["matplotlib.pyplot"] = mock_plt


# Mock llmx TextGenerationConfig
class DummyConfig:
    def __init__(self, n=1, temperature=0, use_cache=True, max_tokens=None):
        self.n = n
        self.temperature = temperature
        self.use_cache = use_cache
        self.max_tokens = max_tokens


sys.modules["llmx"].TextGenerationConfig = DummyConfig

# Import Lida
try:
    from lida.components import Manager
    from lida.domain.models import Summary, Goal
except ImportError as e:
    print(f"Import Error: {e}")
    sys.exit(1)


def verify_structure():
    print("Verifying Lida Manager Structure...")
    lida = Manager(text_gen=MagicMock())

    # Check delegates
    assert lida.lida_app is not None
    assert lida.summarizer_adapter is not None
    assert lida.goal_generator_adapter is not None

    print("Manager structure verified.")


def verify_flow():
    print("Verifying Lida Flow...")
    # Mock TextGenerator in adapters
    mock_text_gen = MagicMock()
    # Mock generation for Goal
    mock_text_gen.generate.return_value.text = [{"content": '[{"index": 0, "question": "q", "visualization": "v", "rationale": "r"}]'}]

    lida = Manager(text_gen=mock_text_gen)
    # Inject behavior for VizGenerator (returns code)
    # We need to distinguish Goal call vs Viz call if using same text_gen mock
    # Or just mock the adapters directly on the LidaApp if possible, but Manager inits them.
    # We will just run visualize and expect it to try to execute.

    summary = Summary(
        name="test",
        file_name="test.csv",
        dataset_description="d",
        field_names=[],
        fields=[],
    )
    goal = Goal(question="q", visualization="v", rationale="r")

    # Mock VizGeneratorAdapter.generate to return code
    lida.viz_generator_adapter.generate = MagicMock(return_value=["import matplotlib.pyplot as plt\ndef plot(data):\n    return plt\nchart = plot(data)"])

    # Run visualize
    charts = lida.visualize(summary=summary, goal=goal, library="seaborn")

    print(f"Visualize returned: {len(charts)} charts")
    if len(charts) > 0:
        print(f"Status: {charts[0].status}")
        print(f"Raster len: {len(charts[0].raster) if charts[0].raster else 0}")
        if charts[0].error:
            print(f"Error: {charts[0].error}")

    assert len(charts) > 0
    assert charts[0].status is True
    assert len(charts[0].raster) > 0
    print("Flow verified.")


if __name__ == "__main__":
    verify_structure()
    verify_flow()
