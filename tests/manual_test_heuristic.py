from lida.datamodel import GoalWebRequest, Summary


def test_heuristic_goal_generation():
    summary = Summary(
        name="cars.csv",
        file_name="cars.csv",
        dataset_description="A dataset about cars",
        field_names=["Miles_per_Gallon", "Cylinders", "Origin"],
        fields=[
            {"column": "Miles_per_Gallon", "properties": {"dtype": "number"}},
            {"column": "Cylinders", "properties": {"dtype": "number"}},
            {
                "column": "Origin",
                "properties": {"dtype": "string", "num_unique_values": 3},
            },
        ],
    )

    GoalWebRequest(summary=summary, n=3, method="heuristic")

    # We can test this by importing Manager directly if we don't want to rely on the running server
    # or we can mock the request to the server if it's running.
    # Since I'm in the environment, I'll instantiate Manager and call goals() directly.

    from lida.components.manager import Manager
    from unittest.mock import MagicMock

    mock_text_gen = MagicMock()
    lida = Manager(text_gen=mock_text_gen)
    goals = lida.goals(summary, n=3, method="heuristic")

    print(f"Generated {len(goals)} goals using heuristic method.")
    for i, goal in enumerate(goals):
        print(f"Goal {i}: {goal.question} ({goal.visualization})")

    assert len(goals) > 0
    assert "histogram" in goals[0].visualization or "scatter" in goals[0].visualization


if __name__ == "__main__":
    test_heuristic_goal_generation()
