import os
import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

# Ensure environment variables are set before importing app
os.environ["GEMINI_API_KEY"] = "fake_key"
os.environ["GOOGLE_API_KEY"] = "fake_key"

from lida.web.app import app

client = TestClient(app)


@pytest.fixture
def mock_lida():
    with patch("lida.web.app.lida") as mock:
        # Configuration matches what api expects
        # Visualization
        mock.visualize.return_value = [{"raster": "base64encoded", "spec": {}, "status": True}]
        mock.edit.return_value = [{"raster": "base64encoded", "spec": {}, "status": True}]
        mock.repair.return_value = [{"raster": "base64encoded", "spec": {}, "status": True}]
        mock.recommend.return_value = [{"raster": "base64encoded", "spec": {}, "status": True}]

        # Explanation
        mock.explain.return_value = [[{"section": "overview", "text": "explanation"}]]

        # Evaluation
        mock.evaluate.return_value = [[{"dimension": "reliability", "score": 10}]]

        # Goals
        # Determine if Goal is an object or dict. app.py returns "data": goals
        # Usually list of Goal objects.
        mock_goal = MagicMock()
        mock_goal.question = "Test Question"
        mock_goal.visualization = "bar chart"
        mock_goal.rationale = "rationale"
        mock_goal.index = 0
        mock.goals.return_value = [mock_goal]

        # Summary
        # app.py expects lida.summarize -> summary object
        mock_summary = MagicMock()
        mock_summary.name = "test_data.csv"
        mock_summary.file_name = "test_data.csv"
        mock_summary.dataset_description = "desc"
        mock_summary.fields = []
        mock_summary.field_names = ["col1"]
        mock.summarize.return_value = mock_summary

        # Infographics
        mock.infographics.return_value = {"images": ["base64encoded"]}

        yield mock


@pytest.fixture
def mock_textgen():
    with patch("lida.web.app.textgen") as mock:
        mock_response = MagicMock()
        mock_response.text = ["Generated text"]
        mock.generate.return_value = mock_response
        yield mock


def test_list_models():
    """Test the /models endpoint"""
    response = client.get("/api/models")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] is True
    assert "data" in data


def test_generate_text(mock_textgen):
    """Test text generation endpoint"""
    payload = {
        "n": 1,
        "temperature": 0.5,
        "model": "gemini-3-flash-preview",
        "provider": "google",
    }
    response = client.post("/api/text/generate", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] is True
    assert "completions" in data


def test_visualize_goals(mock_lida):
    """Test goal generation"""
    payload = {
        "summary": {
            "name": "test_data.csv",
            "file_name": "test_data.csv",
            "dataset_description": "A test dataset",
            "fields": [],
            "field_names": ["col1", "col2"],
        },
        "n": 2,
        "textgen_config": {
            "n": 1,
            "temperature": 0,
            "model": "gemini-3-flash-preview",
            "provider": "google",
        },
    }
    response = client.post("/api/goal", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] is True
    assert len(data["data"]) > 0


def test_visualize_data(mock_lida):
    """Test visualization generation"""
    payload = {
        "summary": {
            "name": "test_data.csv",
            "file_name": "test_data.csv",
            "dataset_description": "A test dataset",
            "fields": [],
            "field_names": ["col1", "col2"],
        },
        "goal": {
            "question": "What is the distribution of col1?",
            "visualization": "histogram",
            "rationale": "To see the spread",
            "index": 0,
        },
        "library": "seaborn",
        "textgen_config": {
            "n": 1,
            "model": "gemini-3-flash-preview",
            "provider": "google",
        },
    }
    response = client.post("/api/visualize", json=payload)
    assert response.status_code == 200
    # app.py returns {"status": True, "charts": ...}
    data = response.json()
    assert data["status"] is True
    assert "charts" in data


def test_visualize_edit(mock_lida):
    """Test visualization editing"""
    payload = {
        "summary": {
            "name": "test_data.csv",
            "file_name": "test_data.csv",
            "dataset_description": "A test dataset",
            "fields": [],
            "field_names": ["col1", "col2"],
        },
        "code": "import matplotlib.pyplot as plt\nplt.plot([1,2,3])",
        "instructions": "Change color to red",
        "library": "matplotlib",
        "textgen_config": {
            "n": 1,
            "model": "gemini-3-flash-preview",
            "provider": "google",
        },
    }
    response = client.post("/api/visualize/edit", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] is True
    assert "charts" in data


def test_visualize_explain(mock_lida):
    """Test visualization explanation"""
    payload = {
        "code": "import matplotlib.pyplot as plt\nplt.plot([1,2,3])",
        "library": "matplotlib",
        "textgen_config": {
            "n": 1,
            "model": "gemini-3-flash-preview",
            "provider": "google",
        },
    }
    response = client.post("/api/visualize/explain", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] is True
    assert "explanations" in data


def test_visualize_evaluate(mock_lida):
    """Test visualization evaluation"""
    payload = {
        "code": "import matplotlib.pyplot as plt\nplt.plot([1,2,3])",
        "goal": {
            "question": "Plot something",
            "visualization": "line chart",
            "rationale": "...",
            "index": 0,
        },
        "library": "matplotlib",
        "textgen_config": {
            "n": 1,
            "model": "gemini-3-flash-preview",
            "provider": "google",
        },
    }
    response = client.post("/api/visualize/evaluate", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] is True
    assert "evaluations" in data


def test_visualize_recommend(mock_lida):
    """Test visualization recommendation"""
    payload = {
        "summary": {
            "name": "test_data.csv",
            "file_name": "test_data.csv",
            "dataset_description": "A test dataset",
            "fields": [],
            "field_names": ["col1", "col2"],
        },
        "code": "import matplotlib.pyplot as plt\nplt.plot([1,2,3])",
        "library": "matplotlib",
        "textgen_config": {
            "n": 1,
            "model": "gemini-3-flash-preview",
            "provider": "google",
        },
    }
    response = client.post("/api/visualize/recommend", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] is True
    assert "charts" in data


def test_visualize_repair(mock_lida):
    """Test visualization repair"""
    payload = {
        "feedback": "Fix the syntax error",
        "code": "import matplotlib.pyplot as plt\nplt.plot([1,2,3)",
        "goal": {
            "question": "Plot something",
            "visualization": "line chart",
            "rationale": "...",
            "index": 0,
        },
        "summary": {
            "name": "test_data.csv",
            "file_name": "test_data.csv",
            "dataset_description": "A test dataset",
            "fields": [],
            "field_names": ["col1", "col2"],
        },
        "library": "matplotlib",
        "textgen_config": {
            "n": 1,
            "model": "gemini-3-flash-preview",
            "provider": "google",
        },
    }
    response = client.post("/api/visualize/repair", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] is True
    assert "charts" in data


def test_summarize_upload(tmp_path, mock_lida):
    """Test file upload summarization"""
    # Create a dummy CSV file
    d = tmp_path / "data.csv"
    d.write_text("col1,col2\n1,2\n3,4")

    with open(d, "rb") as f:
        # The endpoint expects 'file' in multipart/form-data
        response = client.post("/api/summarize", files={"file": ("data.csv", f, "text/csv")})

    assert response.status_code == 200
    data = response.json()
    assert data["status"] is True
    assert "summary" in data


def test_infographics(mock_lida):
    """Test infographics generation"""
    payload = {"visualization": "base64encodedimage", "n": 1, "style_prompt": "pop art"}
    response = client.post("/api/infographer", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] is True
    assert "result" in data
