# LIDA: Automatic Generation of Visualizations and Infographics using Large Language Models

[![PyPI version](https://badge.fury.io/py/lida.svg)](https://badge.fury.io/py/lida)
[![arXiv](https://img.shields.io/badge/arXiv-2303.02927-<COLOR>.svg)](https://arxiv.org/abs/2303.02927)
<a target="_blank" href="https://colab.research.google.com/github/microsoft/lida/blob/main/notebooks/tutorial.ipynb">
<img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

<!-- <img src="docs/images/lidascreen.png" width="100%" /> -->

![lida components](docs/images/lidamodules.jpg)

Details on the components of LIDA are described in the [paper here](https://arxiv.org/abs/2303.02927) and in this tutorial [notebook](notebooks/tutorial.ipynb).

LIDA uses off-the-shelf large language models (OpenAI, PaLM, Cohere, Huggingface) to generate grammar-agnostic visualization specifications and data-faithful infographics.

> **Note:**
> To create visualizations, LIDA _generates_ and _executes_ code.
> Ensure that you run LIDA in a secure environment.

## How it Works - Features Overview

LIDA comprises of 4 modules - A SUMMARIZER that converts data into a rich but compact natural language summary, a GOAL EXPLORER that enumerates visualization goals given the data, a VISGENERATOR that generates, refines, executes and filters visualization code and an INFOGRAPHER module (tbd) that yields data-faithful stylized graphics using IGMs. LIDA provides a python api, and a hybrid user interface (direct manipulation and **multilingual** natural language) for interactive chart, infographics and data story generation. A summary of what you can do with LIDA is listed below:

- **Data Summary**: Given a dataset, generate a compact summary of the data.

```python
from lida.modules import Manager

lida = Manager()
summary = lida.summarize("data/cars.json") # generate data summary
```

- **Goal Generation**: Generate a set of visualization goals given a data summary.

```python
goals = lida.generate_goals(summary, n=5) # generate goals
```

- **Visualization Generation**: Generate, refine, execute and filter visualization code given a data summary and visualization goal. Note that LIDA represents **visualizations as code**.

```python
# generate code specifications for charts
vis_specs = lida.generate_viz(summary=summary, goal=goals[0], library="matplotlib") # seaborn, ggplot ..

# execute code
charts = lida.execute_viz(code_specs=vis_specs)
```

- **Visualization Editing**: Given a visualization, edit the visualization using natural language.

```python
# modify chart using natural language
instructions = ["convert this to a bar chart", "change the color to red", "change y axes label to Fuel Efficiency"]
vis_specs = lida.edit_viz(code=charts[0].code,  summary=summary, instructions=instructions)
edited_chartspecs = lida.execute_viz(code_specs=vis_specs, data=manager.data)

```

- **Visualization Explanation**: Given a visualization, generate a natural language explanation of the visualization code (accessibility, data transformations applied, visualization code)

```python
# generate explanation for chart
explanation = lida.explain_viz(code=charts[0].code, summary=summary)
```

- **Visualization Evaluation and Repair**: Given a visualization, evaluate to find repair instructions (which may be human authored, or generated), repair the visualization.

## Requirements and Installation

**Verify Environment - Python 3.10+**.
Setup and verify that your python environment is `python 3.10` or higher (preferably, use [Conda](https://docs.conda.io/en/main/miniconda.html#installing)).

Once requirements are met, setup your api key and run the following command to install the library in the repository root:

Setup your openai api key. Learn more about setting up keys for other LLM providers [here](https://github.com/victordibia/llmx).

```bash
export OPENAI_API_KEY=<your key>
```

```bash
pip install lida
```

Alternatively you can install the library in dev model by cloning this repo and running `pip install -e .` in the repository root.

## Getting Started

The fastest and recommended way to get started after installation will be to try out the web ui or run the [tutorial notebook](notebooks/tutorial.ipynb).

### Web UI

You can use the library from the bundled ui by running the following command:

```bash
lida ui  --port=8080
```

Then navigate to http://localhost:8080/ in your browser.

Finally, you can call lida from your application via its web api. To view the web api specification, navigate to `http://localhost:8080/api/docs` in your browser.

## Documentation and Citation

A short paper describing LIDA (Accepted at ACL 2023 Conference) is available [here](https://arxiv.org/abs/2303.02927).

```bibtex
@article{dibia2023lida,
      title={LIDA: A Tool for Automatic Generation of Grammar-Agnostic Visualizations and Infographics using Large Language Models},
      author={Victor Dibia},
      year={2023},
      eprint={2303.02927},
      archivePrefix={arXiv},
      primaryClass={cs.AI}
}
```

LIDA builds on insights in automatic generation of visualizaiton from an earlier paper - [Data2Vis: Automatic Generation of Data Visualizations Using Sequence to Sequence Recurrent Neural Networks](https://arxiv.org/abs/1804.03126).
