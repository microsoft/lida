from lida import TextGenerationConfig, llm
from lida.components import Manager
import os
lida = Manager(text_gen=llm("openai"))


cars_data_url = "https://raw.githubusercontent.com/uwdata/draco/master/data/cars.csv"


def test_summarizer():
    textgen_config = TextGenerationConfig(n=1, temperature=0, use_cache=False, max_tokens=None)
    summary_no_enrich = lida.summarize(
        cars_data_url,
        textgen_config=textgen_config,
        summary_method="default")
    summary_enrich = lida.summarize(cars_data_url,
                                    textgen_config=textgen_config, summary_method="llm")

    assert summary_no_enrich != summary_enrich
    assert "dataset_description" in summary_enrich and len(
        summary_enrich["dataset_description"]) > 0


def test_goals():
    textgen_config = TextGenerationConfig(n=1, temperature=0.1, use_cache=False, max_tokens=None)
    summary = lida.summarize(
        cars_data_url,
        textgen_config=textgen_config, summary_method="default")

    goals = lida.goals(summary, n=2, textgen_config=textgen_config)
    assert len(goals) == 2
    assert len(goals[0].question) > 0


def test_vizgen():
    textgen_config = TextGenerationConfig(
        n=1,
        temperature=0.1,
        use_cache=True,
        max_tokens=None)
    summary = lida.summarize(
        cars_data_url,
        textgen_config=textgen_config, summary_method="default")

    goals = lida.goals(summary, n=2, textgen_config=textgen_config)
    charts = lida.visualize(
        summary=summary,
        goal=goals[0],
        textgen_config=textgen_config,
        library="seaborn")

    assert len(charts) > 0
    first_chart = charts[0]

    # Ensure the first chart has a status of True
    assert first_chart.status is True

    # Ensure no errors in the first chart
    assert first_chart.error is None

    # Ensure the raster image of the first chart exists
    assert len(first_chart.raster) > 0

    # Test saving the raster image of the first chart
    temp_file_path = "temp_image.png"
    first_chart.savefig(temp_file_path)
    # Ensure the image is saved correctly
    assert os.path.exists(temp_file_path)
    # Clean up
    os.remove(temp_file_path)