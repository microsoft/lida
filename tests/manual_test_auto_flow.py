import os
import pandas as pd
from unittest.mock import MagicMock

# Import Manager from testing context (needs path setup usually, but we are in root)
from lida.components.manager import Manager


def test_auto_viz_flow():
    # Setup - Create a dummy csv with rich data for advanced plots
    # We need 3 numeric and 2 categorical to trigger everything
    df = pd.DataFrame(
        {
            "A": [1, 2, 3, 4, 5, 2, 3, 4, 5, 1],
            "B": [10, 20, 30, 40, 50, 20, 30, 40, 50, 10],
            "C": [100, 200, 300, 400, 500, 200, 300, 400, 500, 100],
            "Cat1": ["x", "x", "y", "y", "z", "x", "y", "z", "x", "y"],
            "Cat2": [
                "blue",
                "red",
                "blue",
                "red",
                "blue",
                "red",
                "blue",
                "red",
                "blue",
                "red",
            ],
        }
    )
    df.to_csv("test_data.csv", index=False)

    # Mock text gen
    mock_text_gen = MagicMock()

    # Mock enrichment response
    fields_json = '[{"column": "A", "properties": {"dtype": "number"}}, {"column": "B", "properties": {"dtype": "number"}}, {"column": "C", "properties": {"dtype": "number"}}, {"column": "Cat1", "properties": {"dtype": "string", "num_unique_values": 3}}, {"column": "Cat2", "properties": {"dtype": "string", "num_unique_values": 2}}]'
    mock_text_gen.generate.return_value.text = [{"content": '{"dataset_description": "test data", "fields": ' + fields_json + "} "}]

    lida = Manager(text_gen=mock_text_gen)

    # 1. Summarize
    summary = lida.summarize("test_data.csv", summary_method="llm")

    # 2. Heuristic Goals
    goals = lida.goals(summary, n=50, method="heuristic")  # Request many to get all types
    print(f"Goals generated: {len(goals)}")

    # Check for specific advanced types
    viz_types = [g.visualization for g in goals]
    for vt in viz_types:
        print(f"Goal: {vt}")

    has_bubble = any("bubble chart" in vt for vt in viz_types)
    has_heatmap = any("correlation heatmap" in vt for vt in viz_types)
    has_hue_scatter = any("colored by" in vt for vt in viz_types)
    has_box_hue = any("grouped by" in vt for vt in viz_types)
    has_stack = any("stacked bar" in vt for vt in viz_types)

    print("\nCaught Advanced Types:")
    print(f"Bubble: {has_bubble}")
    print(f"Heatmap: {has_heatmap}")
    print(f"Hue Scatter: {has_hue_scatter}")
    print(f"Grouped Box: {has_box_hue}")
    print(f"Stacked Bar: {has_stack}")

    assert has_bubble
    assert has_heatmap
    assert has_hue_scatter
    assert has_box_hue
    assert has_stack

    # 3. Heuristic Viz & Execute (Sample one of each advanced type)
    advanced_goals = [g for g in goals if "bubble" in g.visualization or "heatmap" in g.visualization or "colored by" in g.visualization or "grouped by" in g.visualization or "stacked" in g.visualization]

    # Filter to unique types to test execution
    tested_types = set()
    for goal in advanced_goals:
        vtype = goal.visualization.split(" of ")[0]
        if vtype not in tested_types:
            code_list = lida.visualize(summary, goal, library="seaborn", method="heuristic")
            print(f"\nExecuting {vtype} for {goal.visualization}...")
            # print(code_list[0])
            executed = lida.execute(code_list, df, summary, library="seaborn")
            if executed:
                print(f"Result: {executed[0].status}")
                if not executed[0].status:
                    print(f"Error: {executed[0].error}")
            tested_types.add(vtype)

    # Cleanup
    if os.path.exists("test_data.csv"):
        os.remove("test_data.csv")


if __name__ == "__main__":
    test_auto_viz_flow()
