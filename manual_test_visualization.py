import requests


def test_visualization():
    # Helper to clean up previous run? No need.

    # Prerequisite: Summarize a file first to get summary structure?
    # Or mock the summary.
    # The summary structure is:
    # { "name": ..., "file_name": ..., "fields": [...], "field_names": [...] }

    # Using the file usage from user metadata: uploaded_image_... but real usage uses CSV.
    # Let's assume there is a file "uploaded_image_1767044345121.png" which actually works?
    # User said "we got code again" for the Goal "What is the frequency distribution...".
    # This implies there IS a dataset active.

    # I will try to upload a dummy CSV first to ensure valid state.

    dummy_csv = "col1,spectral_radius,col3\n1,0.5,a\n2,1.2,b\n3,0.9,c"
    with open("test_data.csv", "w") as f:
        f.write(dummy_csv)

    print("Uploading test_data.csv...")
    files = {"file": open("test_data.csv", "rb")}
    try:
        res = requests.post("http://localhost:8080/api/summarize", files=files)
        print(f"Summarize Response: {res.status_code}")
        if res.status_code != 200:
            print(res.text)
            return

        summary = res.json()["summary"]

        # Now visualize
        payload = {
            "summary": summary,
            "goal": {
                "question": "What is the frequency distribution of spectral_radius values?",
                "visualization": "",
                "rationale": "",
            },
            "library": "seaborn",
            "textgen_config": {
                "n": 1,
                "model": "gemini-3-flash-preview",
                "temperature": 0,
            },
        }

        print("Requesting visualization...")
        res_viz = requests.post("http://localhost:8080/api/visualize", json=payload)
        print(f"Visualize Response Code: {res_viz.status_code}")
        print(f"Visualize Response Body: {res_viz.text[:500]}...")  # Print first 500 chars

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    test_visualization()
