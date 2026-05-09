from typing import List
import itertools
import logging
from lida.datamodel import Goal, Summary, Persona

logger = logging.getLogger("lida")


class HeuristicGoalExplorer:
    """Generate goals based on heuristic rules (Grammar of Graphics) without an LLM."""

    def __init__(self) -> None:
        pass

    def generate(self, summary: Summary, n: int = 5, persona: Persona = None) -> List[Goal]:
        """Generate goals given a summary of data using rules."""

        goals = []
        fields = summary.fields if summary.fields else []

        # Classify fields by type
        numeric_fields = [f for f in fields if f["properties"].get("dtype") == "number"]
        categorical_fields = [f for f in fields if f["properties"].get("dtype") in ["category", "string", "boolean"]]
        date_fields = [f for f in fields if f["properties"].get("dtype") == "date"]

        # Rule 1: Univariate - Numeric Distribution (Histogram)
        for field in numeric_fields:
            col = field["column"]
            goals.append(
                Goal(
                    question=f"What is the distribution of {col}?",
                    visualization=f"histogram of {col}",
                    rationale=f"Visualize the distribution of {col} to see the spread and skew of values.",
                    index=len(goals),
                )
            )

        # Rule 2: Univariate - Categorical Count (Bar Chart)
        for field in categorical_fields:
            if field["properties"].get("num_unique_values", 100) < 20:  # Only if readable
                col = field["column"]
                goals.append(
                    Goal(
                        question=f"What is the frequency of each category in {col}?",
                        visualization=f"bar chart of {col}",
                        rationale=f"Determine which categories in {col} are most frequent.",
                        index=len(goals),
                    )
                )

        # Rule 3: Time Series (Line Chart)
        if date_fields and numeric_fields:
            date_col = date_fields[0]["column"]
            for num_col in numeric_fields[:2]:  # Limit to first 2 numeric fields to avoid explosion
                col = num_col["column"]
                goals.append(
                    Goal(
                        question=f"How does {col} change over time?",
                        visualization=f"line chart of {col} by {date_col}",
                        rationale=f"Observe trends in {col} over {date_col}.",
                        index=len(goals),
                    )
                )

        # Rule 4: Bivariate - Correlation (Scatter Plot) - Combinatorial
        # "Take all the headings and produce a scatter plot for each pair of headings."
        if len(numeric_fields) >= 2:
            # Generate all unique pairs
            for col1_field, col2_field in itertools.combinations(numeric_fields, 2):
                col1 = col1_field["column"]
                col2 = col2_field["column"]
                goals.append(
                    Goal(
                        question=f"What is the relationship between {col1} and {col2}?",
                        visualization=f"scatter plot of {col1} vs {col2}",
                        rationale=f"Investigate if there is a correlation between {col1} and {col2}.",
                        index=len(goals),
                    )
                )

        # Rule 5: Bivariate - Categorical vs Numeric (Bar Chart / Box Plot)
        if categorical_fields and numeric_fields:
            if len(categorical_fields) > 0 and len(numeric_fields) > 0:
                cat_col = categorical_fields[0]["column"]
                num_col = numeric_fields[0]["column"]
                if categorical_fields[0]["properties"].get("num_unique_values", 100) < 20:
                    goals.append(
                        Goal(
                            question=f"How does {num_col} vary by {cat_col}?",
                            visualization=f"bar chart of {num_col} by {cat_col}",
                            rationale=f"Compare the average {num_col} across different {cat_col} groups.",
                            index=len(goals),
                        )
                    )

        # Rule 6: Multivariate Scatter (Hue) - 2 Numeric + 1 Categorical
        if len(numeric_fields) >= 2 and len(categorical_fields) >= 1:
            col1 = numeric_fields[0]["column"]
            col2 = numeric_fields[1]["column"]
            cat_col = categorical_fields[0]["column"]
            if categorical_fields[0]["properties"].get("num_unique_values", 100) < 10:  # Limit colors
                goals.append(
                    Goal(
                        question=f"What is the relationship between {col1} and {col2}, colored by {cat_col}?",
                        visualization=f"scatter plot of {col1} vs {col2} colored by {cat_col}",
                        rationale=f"Visualize correlation between {col1} and {col2} with {cat_col} context.",
                        index=len(goals),
                    )
                )

        # Rule 7: Bubble Chart - 3 Numeric
        if len(numeric_fields) >= 3:
            col1 = numeric_fields[0]["column"]
            col2 = numeric_fields[1]["column"]
            col3 = numeric_fields[2]["column"]
            goals.append(
                Goal(
                    question=f"What is the relationship between {col1}, {col2} (size: {col3})?",
                    visualization=f"bubble chart of {col1} vs {col2} sized by {col3}",
                    rationale=f"Explore the interaction of {col1}, {col2} and {col3}.",
                    index=len(goals),
                )
            )

        # Rule 8: Global Correlation Heatmap - 3+ Numeric
        if len(numeric_fields) >= 3:
            goals.append(
                Goal(
                    question="What are the correlations between all numeric variables?",
                    visualization="correlation heatmap of numeric variables",
                    rationale="Overview of all pairwise correlations.",
                    index=len(goals),
                )
            )

        # Rule 9: Nested Box Plot - 1 Numeric + 2 Categorical
        if len(numeric_fields) >= 1 and len(categorical_fields) >= 2:
            num_col = numeric_fields[0]["column"]
            cat1 = categorical_fields[0]["column"]
            cat2 = categorical_fields[1]["column"]
            if categorical_fields[1]["properties"].get("num_unique_values", 100) < 10:
                goals.append(
                    Goal(
                        question=f"Distribution of {num_col} by {cat1} grouped by {cat2}?",
                        visualization=f"box plot of {num_col} by {cat1} grouped by {cat2}",
                        rationale="Detailed distribution analysis across subgroups.",
                        index=len(goals),
                    )
                )

        # Rule 10: Stacked Bar Chart - 2 Categorical
        if len(categorical_fields) >= 2:
            cat1 = categorical_fields[0]["column"]
            cat2 = categorical_fields[1]["column"]
            if categorical_fields[0]["properties"].get("num_unique_values", 100) < 20 and categorical_fields[1]["properties"].get("num_unique_values", 100) < 10:
                goals.append(
                    Goal(
                        question=f"Count of {cat1} stacked by {cat2}?",
                        visualization=f"stacked bar chart of {cat1} stack by {cat2}",
                        rationale=f"Compare proportions of {cat2} within {cat1}.",
                        index=len(goals),
                    )
                )

        # Shuffle or prioritize? For now, just slice
        # Ideally we score them, but this is a heuristic "dumber" version.
        return goals[:n]
