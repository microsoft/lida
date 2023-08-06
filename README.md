# LIDA: Automatic Generation of Visualizations and Infographics using Large Language Models

<!-- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](#) -->
<!-- image  -->

LIDA uses off-the-shelf large language models to generate grammar-agnostic visualization specifications and data-faithful infographics.

> **Important Note on Code Execution**: To generate visualizations, LIDA _generates_ and _executes_ code. Ensure that you run LIDA in a secure environment.

## How it works

LIDA comprises of 4 modules - A SUMMARIZER that converts data into a rich but compact natural language summary, a GOAL EXPLORER that enumerates visualization goals given the data, a VISGENERATOR that generates, refines, executes and filters visualization code and an INFOGRAPHER module that yields data-faithful stylized graphics using IGMs. LIDA provides a python api, and a hybrid user interface (direct manipulation and **multilingual** natural language) for interactive chart, infographics and data story generation.

![lida components](docs/images/lidamodules.jpg)

Details on the components of LIDA are described in the [paper here](https://arxiv.org/abs/2303.02927).

## Requirements and Installation

**Verify Environment - Python 3.10+**.
Setup and verify that your python environment is `python 3.10` or higher (preferably, use Conda).

Once requirements are met, run the following command to install the library in the repository root:

Setup your openai api key

```bash
export OPENAI_API_KEY=<your key>
```

```bash
pip install -e .
```

## Python API

LIDA provides a `python` api for generating visualizations and infographics - data summary generation, visualization goals, visualization generation, visualization editing.

### Data Summarization

```python
from lida.modules import Manager

lida = Manager()
summary = lida.summarize("data/cars.json") # generate data summary
```

### Visualization Goal Generation

```python
goals = lida.generate_goals(summary, n=5) # generate goals
```

### Visualization Generation

```python
# generate altair charts
vis_specs = lida.generate_viz(summary=summary, goal=goals[0], library="matplotlib") # altair, matplotlib etc

# execute charts - returns altair chart objects
charts = lida.execute_viz(code_specs=vis_specs)
```

### Visualization Editing

```python
# modify chart using natural language
instructions = ["convert this to a bar chart", "change the color to red", "change y axes label to Fuel Efficiency"]
vis_specs = lida.edit_viz(code=charts[0].code,  summary=summary, instructions=instructions, library="matplotlib")
edited_chartspecs = lida.execute_viz(code_specs=vis_specs, data=manager.data)

```

## Getting Started

The fastest and recommended way to get started after installation will be to try out the web ui or run the [tutorial notebook](notebooks/tutorial.ipynb).

### Web UI

You can use the library from the ui by running the following command:

```bash
lida ui  --port=8080
```

Then navigate to http://localhost:8080/ in your browser.

Finally, you can call lida from your application via its web api. To view the web api specification, navigate to `http://localhost:8080/api/docs` in your browser.

### Tutorial Notebook

Learn more about the python api in this [tutorial notebook](/notebooks/tutorial.ipynb).

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
