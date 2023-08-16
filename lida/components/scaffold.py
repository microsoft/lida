from dataclasses import asdict

from lida.datamodel import Goal


# if len(plt.xticks()[0])) > 20 assuming plot is made with plt or
# len(ax.get_xticks()) > 20 assuming plot is made with ax, set a max of 20
# ticks on x axis, ticker.MaxNLocator(20)

class ChartScaffold(object):
    """Return code scaffold for charts in multiple visualization libraries"""

    def __init__(
        self,
    ) -> None:

        pass

    def get_template(self, goal: Goal, library: str):
        mpl_pre = f"Set chart title to {goal.question}. If the solution requires a single value (e.g. max, min, median, first, last etc), ALWAYS add a line (axvline or axhline) to the chart, ALWAYS with a legend containing the single value (formatted with 0.2F). If using a <field> where semantic_type=date, YOU MUST APPLY the following transform before using that column i) convert date fields to date types using data[''] = pd.to_datetime(data[<field>], errors='coerce'), ALWAYS use  errors='coerce' ii) drop the rows with NaT values data = data[pd.notna(data[<field>])] iii) convert field to right time format for plotting.  ALWAYS make sure the x-axis labels are legible (e.g., rotate when needed). Use BaseMap for charts that require a map. Given the dataset summary, the plot(data) method should generate a {library} chart ({goal.visualization}) that addresses this goal: {goal.question}. DO NOT include plt.show(). Note that data in the plot(data) method is a pandas Dataframe that already contains the data. The plot method must return a matplotlib object. Think step by step. \n"

        if library == "matplotlib":
            instructions = {"role": "assistant", "content": mpl_pre}
            template = \
                f"""
import matplotlib.pyplot as plt
import pandas as pd
<imports>
# plan -
def plot(data: pd.DataFrame):
    <stub> # only modify this section
    plt.title('{goal.question}', wrap=True)
    return plt;

chart = plot(data) # data already contains the data to be plotted. Always include this line"""
        elif library == "seaborn":
            instructions = {"role": "assistant", "content": mpl_pre}

            template = \
                f"""
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
<imports>
# solution plan
# i.  ..
def plot(data: pd.DataFrame):

    <stub> # only modify this section
    plt.title('{goal.question}', wrap=True)
    return plt;

chart = plot(data) # data already contains the data to be plotted. Always include this line"""

        elif library == "ggplot":
            instructions = {
                "role": "assistant",
                "content": f" If using a field where semantic_type=date, do the following i) convert date fields to date types using data[''] = pd.to_datetime(data[''], errors='coerce'), ALWAYS use  errors='coerce' ii) drop the rows with NaT values data = data[pd.notna(data[''])] iii) convert field to right time format for plotting.  ALWAYS make sure the x-axis labels are legible.  Solve the task  carefully by completing ONLY the <imports> AND <stub> section. Given the dataset summary, the plot(data) method should generate a {library} chart ({goal.visualization}) that addresses this goal: {goal.question}. The plot method must return a ggplot object (chart)`. Think step by step.p. \n",
            }

            template = \
                f"""
import plotnine as p9
<imports>
def plot(data: pd.DataFrame):
    chart = <stub>

    return chart;

chart = plot(data) # data already contains the data to be plotted. Always include this line"""

        elif library == "altair":
            instructions = {
                "role": "system",
                "content": f"If using a field where semantic_type=date, do the following i) convert date fields to date types using data[''] = pd.to_datetime(data[''], errors='coerce'), ALWAYS use  errors='coerce' ii) drop the rows with NaT values data = data[pd.notna(data[''])] iii) convert field to right time format for plotting.  ALWAYS make sure the x-axis labels are legible.  Solve the task  carefully by completing ONLY the <imports> AND <stub> section. Given the dataset summary, the plot(data) method should generate a {library} chart ({goal.visualization}) that addresses this goal: {goal.question}. Always add a type that is BASED on semantic_type to each field such as :Q, :O, :N, :T, :G. Use :T if semantic_type is year or date. The plot method must return an altair object (chart)`. Think step by step. \n",
            }
            template = \
                """
import altair as alt
<imports>
def plot(data: pd.DataFrame):
    <stub> # only modify this section
    return chart
chart = plot(data) # data already contains the data to be plotted. Do not modify  this line
"""
        else:
            raise ValueError(
                "Unsupported library. Choose from 'matplotlib', 'seaborn', 'plotly', 'bokeh', 'ggplot', 'altair'."
            )

        return template, instructions
