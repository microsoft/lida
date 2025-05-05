"""
Example demonstrating the LIDA Analyzer with multimodal RAG capabilities.

This example shows how to:
1. Load a dataset
2. Generate a summary
3. Create a visualization
4. Analyze the visualization using RAG
5. Ingest additional documents to enhance the analysis context
"""

import os
import sys
import pandas as pd
from llmx import TextGenerationConfig

# Import LIDA
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lida import Manager

# Set up OpenAI API key - replace with your API key or set environment variable
# os.environ["OPENAI_API_KEY"] = "your-api-key"

# Initialize LIDA Manager
lida_manager = Manager()

def main():
    # Load sample data
    # Replace with your own dataset path
    data_path = "examples/data/sample_data.csv"
    if not os.path.exists(data_path):
        # Create sample data if it doesn't exist
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        df = pd.DataFrame({
            'Product': ['A', 'B', 'C', 'D', 'E'],
            'Sales': [100, 200, 150, 300, 250],
            'Year': [2019, 2020, 2021, 2022, 2023]
        })
        df.to_csv(data_path, index=False)
        print(f"Created sample data at {data_path}")
    
    # 1. Load data and generate summary
    print("Generating data summary...")
    summary = lida_manager.summarize(data=data_path)
    print(f"Generated summary for {summary.name}")
    
    # 2. Generate visualization goals
    print("Generating visualization goals...")
    goals = lida_manager.goals(summary=summary)
    print(f"Generated {len(goals)} goals")
    
    # 3. Generate visualization
    selected_goal = goals[0]  # Select the first goal
    print(f"Selected goal: {selected_goal.question}")
    
    viz_config = TextGenerationConfig(temperature=0.2)
    print("Generating visualization...")
    visualization = lida_manager.visualize(
        summary=summary, 
        goal=selected_goal,
        textgen_config=viz_config,
        library="seaborn"  # Use seaborn to get raster output
    )
    print("Visualization generated")
    
    # 4. Execute the visualization
    print("Executing visualization...")
    executed_viz = lida_manager.execute(
        code_specs=visualization,
        data=data_path,
        summary=summary,
        library="seaborn"
    )
    print("Visualization executed")
    
    # Select the first successful visualization
    for viz in executed_viz:
        if viz.status:
            executed_chart = viz
            break
    else:
        print("No successful visualizations generated")
        return
    
    # 5. Create a documents directory for additional context
    docs_dir = "examples/data/docs"
    os.makedirs(docs_dir, exist_ok=True)
    
    # Create a sample PDF with additional context (as a text file for this example)
    sample_doc_path = os.path.join(docs_dir, "market_trends.txt")
    if not os.path.exists(sample_doc_path):
        with open(sample_doc_path, "w") as f:
            f.write("""
            # Market Trends Report
            
            The sales of Product A have been steadily increasing over the past few years.
            Product B saw a significant jump in 2020, likely due to changing market conditions.
            Product C has maintained consistent performance but is expected to grow in the coming years.
            Product D has outperformed all other products since its introduction.
            Product E is our newest offering and is already showing strong performance.
            """)
    
    # 6. Analyze the visualization using RAG
    print("\nAnalyzing visualization without context...")
    analysis_without_context = lida_manager.analyze(
        chart=executed_chart,
        summary=summary,
        query="What insights can be derived from this visualization?",
    )
    print("\nAnalysis without additional context:")
    print(analysis_without_context)
    
    # 7. Ingest the document to enhance the analysis context
    print("\nIngesting additional context documents...")
    # In a real-world scenario, you would use PDF documents
    # lida_manager.ingest_document("examples/data/docs/market_trends.pdf")
    
    # For this example, we'll simulate adding web content
    lida_manager.ingest_web_content(
        url="https://example.com/market_trends",
        content=open(sample_doc_path).read()
    )
    
    # 8. Re-analyze with the additional context
    print("\nAnalyzing visualization with context...")
    analysis_with_context = lida_manager.analyze(
        chart=executed_chart,
        summary=summary,
        query="What insights can be derived from this visualization in relation to market trends?",
    )
    print("\nAnalysis with additional context:")
    print(analysis_with_context)
    
    # 9. Evaluate the visualization
    print("\nEvaluating the visualization...")
    evaluation = lida_manager.evaluate(
        code=executed_chart.code,
        goal=selected_goal
    )
    print("\nVisualization evaluation:")
    print(evaluation)
    
    print("\nComplete! Example finished successfully.")

if __name__ == "__main__":
    main() 