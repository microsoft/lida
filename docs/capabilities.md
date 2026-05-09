This README provides an overview of the current capabilities of LIDA.

### Core Capabilities

These are the fundamental features that form the primary functionality of LIDA related to visualizations and infographics.

| Core Feature                      | Description                                                             | Status |
| --------------------------------- | ----------------------------------------------------------------------- | ------ |
| Data Summarization                | Generates a compact summary of the data.                                | ✅      |
| Goal Generation                   | Produces a set of visualization goals from a data summary.              | ✅      |
| Visualization Generation          | Creates and executes visualization code based on data summary and goal. | ✅      |
| Visualization Editing             | Modifies visualizations using natural language instructions.            | ✅      |
| Visualization Explanation         | Generates natural language explanations of visualization code.          | ✅      |
| Visualization Evaluation & Repair | Evaluates visualizations and provides repair instructions.              | ✅      |
| Visualization Recommendation      | Recommends a set of visualizations based on a dataset.                  | ✅      |
| Infographic Generation            | Converts visualizations to data-faithful infographics.                  | 🚧      |

> ⚠️ **Note**: LIDA is currently optimized for generating visualizations i.e. tasks for which the output is a visualization. It may not be the best tool for tasks that do not involve visualizations, such as creating machine learning models (e.g., create a time series model for forecasting), data analysis with a single value answer (what is square root of the smallest value in the dataset). This may be supported in the future.

### Other Capabilities

These features support the core capabilities and provide additional utility and flexibility.

| Other Feature                 | Description                                                                      | Status | Notes                                                 |
| ----------------------------- | -------------------------------------------------------------------------------- | ------ | ----------------------------------------------------- |
| Grammar-Agnostic              | Works with any programming language and visualization library.                   | ✅      |                                                       |
| Multi-LLM Provider Support    | Compatible with various large language model providers like OpenAI, Gemini, etc. | ✅      |                                                       |
| Python API                    | Provides a Python-based API for generating visualizations & infographics.        | ✅      | Requires Python 3.10 or higher.                       |
| Web API & UI                  | Optional user interface and web API included for exploration.                    | ✅      | Setup via Docker; accessible via localhost.           |
| Docker Support                | Can be set up and run using Docker.                                              | ✅      | Facilitates deployment and containerization.          |
| HuggingFace Model Integration | Supports using HuggingFace models for text generation.                           | ✅      | User can opt for direct use or via a local endpoint.  |
| Security Note                 | Generates and executes code; should be run in a secure environment.              | ⚠️      | Proper permissions management is crucial.             |
| Community Expansion           | Encourages community contributions and extensions of the tool.                   | ✅      | Examples available, e.g., lida-streamlit.             |
| Documentation & Citation      | Well-documented with available academic paper citation.                          | ✅      | Provides theoretical background and use case details. |

Symbols used:

- ✅ Feature is included and functional.
- 🚧 Feature is still in development or beta stage.
- ⚠️ Feature requires careful handling due to security implications.
