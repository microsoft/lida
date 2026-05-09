import sys
import json
from unittest.mock import MagicMock


def pytest_configure(config):
    # Mock llmx
    mock_llm_instance = MagicMock()

    def generate_side_effect(messages, config):
        system_content = messages[0]["content"] if messages else ""

        # Decide what to return based on prompt
        if "Annotation" in system_content or "Annotate the dictionary" in str(messages) or "updated JSON dictionary" in system_content:
            # Summarizer expects a dict
            content_data = {
                "name": "mock_data.csv",
                "file_name": "mock_data.csv",
                "dataset_description": "Enriched description",
                "fields": [],
                "field_names": [],
            }
            content = json.dumps(content_data)
            mock_resp = MagicMock()
            mock_resp.text = [{"content": "```json\n" + content + "\n```", "role": "assistant"}]
            return mock_resp
        elif "PERFECT code" in system_content:
            # VizGenerator expects code
            content = """
import matplotlib.pyplot as plt
def plot(data):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot([1, 2, 3], [1, 2, 3])
    return plt

chart = plot(data)
"""
            mock_resp = MagicMock()
            mock_resp.text = [{"content": "```python\n" + content + "\n```", "role": "assistant"}]
            return mock_resp
        else:
            # Default to List (Goals, Personas, etc.)
            content_data = [
                {
                    "index": 0,
                    "question": "q1",
                    "visualization": "v1",
                    "rationale": "r1",
                    "persona": "p1",
                },
                {
                    "index": 1,
                    "question": "q2",
                    "visualization": "v2",
                    "rationale": "r2",
                    "persona": "p2",
                },
            ]
            content = json.dumps(content_data)

            mock_resp = MagicMock()
            # Lida utils cleans code snippet looking for backticks
            mock_resp.text = [{"content": "```json\n" + content + "\n```", "role": "assistant"}]
            return mock_resp

    mock_llm_instance.generate.side_effect = generate_side_effect

    llmx_module = MagicMock()
    llmx_module.llm.return_value = mock_llm_instance
    llmx_module.TextGenerator = MagicMock

    from pydantic import BaseModel

    class DummyTextGenerationConfig(BaseModel):
        n: int = 1
        temperature: float = 0
        model: str = None
        provider: str = None
        use_cache: bool = True

    llmx_module.TextGenerationConfig = DummyTextGenerationConfig
    sys.modules["llmx"] = llmx_module

    # Mock peacasso
    peacasso = MagicMock()
    sys.modules["peacasso"] = peacasso
    sys.modules["peacasso.generator"] = MagicMock()
    sys.modules["peacasso.datamodel"] = MagicMock()
    sys.modules["peacasso.utils"] = MagicMock()

    # Mock diskcache, tiktoken
    sys.modules["diskcache"] = MagicMock()
    sys.modules["tiktoken"] = MagicMock()

    # Mock visualization libraries
    sys.modules["plotly"] = MagicMock()
    sys.modules["plotly.io"] = MagicMock()
    sys.modules["plotly.graph_objects"] = MagicMock()
    sys.modules["matplotlib"] = MagicMock()

    class MockPyplot:
        def subplots(self, *args, **kwargs):
            return (MagicMock(), MagicMock())

        def box(self, *args, **kwargs):
            pass

        def grid(self, *args, **kwargs):
            pass

        def savefig(self, *args, **kwargs):
            args[0].write(b"dummy_png_data")

        def close(self, *args, **kwargs):
            pass

        def title(self, *args, **kwargs):
            pass

        def show(self, *args, **kwargs):
            pass

    sys.modules["matplotlib.pyplot"] = MockPyplot()
    sys.modules["seaborn"] = MagicMock()
    sys.modules["altair"] = MagicMock()
    sys.modules["torch"] = MagicMock()

    # Mock ChartExecutor to avoid running code and using matplotlib
    exec_module = MagicMock()
    fake_executor_instance = MagicMock()
    fake_response = MagicMock()
    fake_response.status = True
    fake_response.raster = "fake_raster_data"
    fake_response.error = None

    def savefig_side_effect(path, *args, **kwargs):
        # Create dummy file
        with open(path, "w") as f:
            f.write("mock")

    fake_response.savefig.side_effect = savefig_side_effect

    fake_executor_instance.execute.return_value = [fake_response]
    exec_module.ChartExecutorAdapter.return_value = fake_executor_instance

    sys.modules["lida.infrastructure.adapters.executor"] = exec_module
