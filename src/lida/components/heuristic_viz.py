import re
import logging
from lida.datamodel import Goal, Summary

logger = logging.getLogger("lida")


class HeuristicVizGenerator:
    """Generate visualization code based on heuristic rules without an LLM."""

    def __init__(self) -> None:
        pass

    def generate(self, summary: Summary, goal: Goal, library: str = "seaborn") -> list[str]:
        """Generate visualization code given a summary and a goal."""

        # We only support seaborn for now as it's the most robust for this
        if library != "seaborn":
            logger.warning(f"Heuristic mode only works best with seaborn, but {library} was requested. Defaulting to seaborn-like structure.")

        viz_type = goal.visualization
        code = ""

        # Parsing logic matches HeuristicGoalExplorer outputs

        # 1. Histogram: "histogram of {col}"
        match = re.search(r"histogram of (.+)", viz_type, re.IGNORECASE)
        if match:
            col = match.group(1)
            code = f"""
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

def plot(data: pd.DataFrame):
    plt.figure(figsize=(10, 6))
    sns.histplot(data=data, x='{col}', kde=True)
    plt.title('{goal.question}')
    return plt

chart = plot(data)
"""

        # 2. Bar Chart (Count): "bar chart of {col}" (univariate)
        if not code:
            match = re.search(r"bar chart of (.+)", viz_type, re.IGNORECASE)
            if match and " by " not in viz_type.lower():
                col = match.group(1)
                code = f"""
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

def plot(data: pd.DataFrame):
    plt.figure(figsize=(10, 6))
    order = data['{col}'].value_counts().index[:20] # Limit to top 20
    sns.countplot(data=data, x='{col}', order=order)
    plt.title('{goal.question}')
    plt.xticks(rotation=45)
    return plt

chart = plot(data)
"""

        # 3. Line Chart: "line chart of {col} by {date_col}"
        if not code:
            match = re.search(r"line chart of (.+) by (.+)", viz_type, re.IGNORECASE)
            if match:
                y_col = match.group(1)
                x_col = match.group(2)
                code = f"""
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

def plot(data: pd.DataFrame):
    plt.figure(figsize=(12, 6))
    # Ensure date column is datetime
    data['{x_col}'] = pd.to_datetime(data['{x_col}'], errors='coerce')
    data = data.dropna(subset=['{x_col}'])
    
    sns.lineplot(data=data, x='{x_col}', y='{y_col}')
    plt.title('{goal.question}')
    plt.xticks(rotation=45)
    return plt

chart = plot(data)
"""

        # 6. Multivariate Scatter (Hue): "scatter plot of {col1} vs {col2} colored by {cat_col}"
        if not code:
            match = re.search(r"scatter plot of (.+) vs (.+) colored by (.+)", viz_type, re.IGNORECASE)
            if match:
                col1 = match.group(1)
                col2 = match.group(2)
                cat_col = match.group(3)
                code = f"""
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

def plot(data: pd.DataFrame):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=data, x='{col1}', y='{col2}', hue='{cat_col}')
    plt.title('{goal.question}')
    return plt

chart = plot(data)
"""

        # 7. Bubble Chart: "bubble chart of {col1} vs {col2} sized by {col3}"
        if not code:
            match = re.search(r"bubble chart of (.+) vs (.+) sized by (.+)", viz_type, re.IGNORECASE)
            if match:
                col1 = match.group(1)
                col2 = match.group(2)
                col3 = match.group(3)
                code = f"""
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

def plot(data: pd.DataFrame):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=data, x='{col1}', y='{col2}', size='{col3}', sizes=(20, 200), legend=True)
    plt.title('{goal.question}')
    return plt

chart = plot(data)
"""

        # 4. Scatter Plot: "scatter plot of {col1} vs {col2}"
        if not code:
            match = re.search(r"scatter plot of (.+) vs (.+)", viz_type, re.IGNORECASE)
            if match and "colored by" not in viz_type.lower() and "sized by" not in viz_type.lower():
                col1 = match.group(1)
                col2 = match.group(2)
                code = f"""
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

def plot(data: pd.DataFrame):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=data, x='{col1}', y='{col2}')
    plt.title('{goal.question}')
    return plt

chart = plot(data)
"""

        # 5. Bar Chart (Bivariate): "bar chart of {num_col} by {cat_col}"
        if not code:
            match = re.search(r"bar chart of (.+) by (.+)", viz_type, re.IGNORECASE)
            if match and "grouped by" not in viz_type.lower() and "stack by" not in viz_type.lower():
                val_col = match.group(1)
                cat_col = match.group(2)
                code = f"""
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

def plot(data: pd.DataFrame):
    plt.figure(figsize=(10, 6))
    # Aggregation usually implied for bar chart of num by cat
    sns.barplot(data=data, x='{cat_col}', y='{val_col}')
    plt.title('{goal.question}')
    plt.xticks(rotation=45)
    return plt

chart = plot(data)
"""

        # 8. Correlation Heatmap: "correlation heatmap of numeric variables"
        if "correlation heatmap" in viz_type.lower():
            code = f"""
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot(data: pd.DataFrame):
    plt.figure(figsize=(10, 8))
    numeric_data = data.select_dtypes(include=[np.number])
    corr = numeric_data.corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('{goal.question}')
    return plt

chart = plot(data)
"""

        # 9. Nested Box Plot: "box plot of {num_col} by {cat1} grouped by {cat2}"
        if not code:
            match = re.search(r"box plot of (.+) by (.+) grouped by (.+)", viz_type, re.IGNORECASE)
            if match:
                num_col = match.group(1)
                cat1 = match.group(2)
                cat2 = match.group(3)
                code = f"""
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

def plot(data: pd.DataFrame):
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=data, x='{cat1}', y='{num_col}', hue='{cat2}')
    plt.title('{goal.question}')
    plt.xticks(rotation=45)
    return plt

chart = plot(data)
"""

        # 10. Stacked Bar Chart: "stacked bar chart of {cat1} stack by {cat2}"
        if not code:
            match = re.search(r"stacked bar chart of (.+) stack by (.+)", viz_type, re.IGNORECASE)
            if match:
                cat1 = match.group(1)
                cat2 = match.group(2)
                code = f"""
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

def plot(data: pd.DataFrame):
    plt.figure(figsize=(12, 6))
    # Create crosstab
    ct = pd.crosstab(data['{cat1}'], data['{cat2}'])
    ct.plot(kind='bar', stacked=True)
    plt.title('{goal.question}')
    plt.xlabel('{cat1}')
    plt.xticks(rotation=45)
    plt.tight_layout()
    return plt # returning plt might be tricky with pandas plot? 
    # Actually pandas plot returns axes, but plt.gcf() works.
    
    # Correction: lida expects plt or object with savefig/to_json?
    # Usually we wrap in function and return plt.
    # The 'ct.plot' calls matplotlib under hood.
    return plt

chart = plot(data)
"""

        if not code:
            # Fallback
            logger.warning(f"Could not parse visualization type: {viz_type}")
            code = f"""
import matplotlib.pyplot as plt
import pandas as pd

def plot(data: pd.DataFrame):
    plt.figure()
    plt.text(0.5, 0.5, "Could not generate chart for: {viz_type}", ha='center')
    return plt

chart = plot(data)
"""
        return [code]
