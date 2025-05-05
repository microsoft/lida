# LIDA Analyzer Module

The LIDA Analyzer is a multimodal Retrieval-Augmented Generation (RAG) module that enhances visualization analysis within the LIDA framework. It analyzes visualizations and generates context-aware insights by leveraging various knowledge sources.

## Overview

The Analyzer module adds the following capabilities to LIDA:

1. **Multimodal Analysis**: Analyzes charts by understanding both the visual elements and the code that generated them
2. **Context-Aware Insights**: Enhances analysis with relevant context retrieved from a knowledge base
3. **Knowledge Integration**: Ingests documents, web content, and LIDA-generated summaries into the knowledge base
4. **Seamless Integration**: Works with existing LIDA components like the Summarizer and VisGenerator

## Architecture

The Analyzer consists of the following key components:

### 1. MultimodalRAGPipeline

The core RAG engine that:
- Extracts and processes chart images
- Retrieves relevant context from the knowledge base
- Generates insights using multimodal LLMs

### 2. Vector Database

Stores embeddings of:
- Document chunks from PDFs and web content
- Data summaries from LIDA's Summarizer
- Previous analysis results

### 3. Knowledge Management

Utilities for:
- Ingesting and chunking documents
- Updating the knowledge base with new information
- Maintaining the vector store

## Usage

### Basic Usage

To analyze a chart with the Analyzer:

```python
from lida import Manager

# Initialize LIDA Manager
lida_manager = Manager()

# Load data and create visualization (standard LIDA workflow)
summary = lida_manager.summarize(data="path/to/data.csv")
goals = lida_manager.goals(summary=summary)
visualization = lida_manager.visualize(summary=summary, goal=goals[0])
executed_viz = lida_manager.execute(code_specs=visualization, data="path/to/data.csv", summary=summary)

# Analyze the chart with RAG
analysis = lida_manager.analyze(
    chart=executed_viz[0],
    summary=summary,
    query="What insights can be derived from this visualization?"
)

print(analysis)
```

### Adding Context

To enhance the analysis with additional context:

```python
# Ingest a PDF document
lida_manager.ingest_document("path/to/document.pdf")

# Ingest web content
lida_manager.ingest_web_content(
    url="https://example.com/article",
    content="Content of the article..."
)

# Update the knowledge base
lida_manager.update_knowledge_base("path/to/documents/directory")

# Analyze with the enhanced context
enhanced_analysis = lida_manager.analyze(
    chart=executed_viz[0],
    summary=summary,
    query="What insights can be derived from this visualization in relation to industry trends?"
)
```

## Configuration

The Analyzer can be configured with different embedding models, vector databases, and LLMs:

```python
from lida.components.analysis import Analyzer, MultimodalRAGPipeline

# Custom configuration
config = {
    "embedding_provider": "openai",
    "embedding_kwargs": {"model": "text-embedding-3-small"},
    "vector_db_provider": "milvus",
    "vector_db_kwargs": {
        "connection_args": {"host": "localhost", "port": "19530"},
        "collection_name": "lida_knowledge"
    },
    "llm_provider": "google",
    "llm_kwargs": {"model": "gemini-1.5-pro-latest", "temperature": 0.2}
}

# Initialize with custom configuration
custom_pipeline = MultimodalRAGPipeline(**config)
custom_analyzer = Analyzer()
custom_analyzer.rag_pipeline = custom_pipeline
```

## Advanced Features

### 1. Chart Image Processing

The Analyzer can process both raster images (from libraries like Matplotlib/Seaborn) and vector specifications (from libraries like Altair/Vega-Lite).

### 2. Incremental Knowledge Updates

The system can detect changes in source documents and incrementally update the knowledge base without reprocessing all content.

### 3. Cross-modal Retrieval

The Analyzer can retrieve relevant context across different modalities (text, images, tables) to provide comprehensive insights.

## Integration Points with LIDA

The Analyzer integrates with LIDA through the following touchpoints:

1. Accesses visualization outputs from the `execute()` method
2. Uses data summaries from the `summarize()` method
3. Extends the Manager class with new methods for document ingestion and analysis
4. Complements the VizExplainer by adding context-aware insights

## Requirements

The Analyzer requires the following dependencies (already included in the updated requirements.txt):

- langchain>=0.1.0
- langchain-openai>=0.0.5 (for OpenAI models)
- langchain-google-genai>=0.0.5 (for Google Gemini models)
- faiss-cpu>=1.7.4 (for vector storage)
- pymilvus>=2.3.0 (optional, for Milvus vector database)
- pypdf>=3.17.0 (for PDF document processing)

## Limitations and Future Work

- Currently supports PDF documents and plain text web content
- Image rendering for vector-based charts needs a secure execution environment
- Future versions will support more document types and richer multimodal content

## References

- [LangChain MultiVector Retriever Documentation](https://python.langchain.com/docs/modules/data_connection/retrievers/multi_vector)
- [RAG Best Practices](https://python.langchain.com/docs/use_cases/question_answering/rag_with_guidelines)
- [ChartQA Benchmark](https://github.com/vis-nlp/chart-question-answering) 