import base64
import io
import logging
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from langchain.chains import LLMChain
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import CohereEmbeddings, HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.retrievers import MultiVectorRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.schema import Document
from langchain.schema.runnable import RunnablePassthrough
from langchain.storage import InMemoryStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS, Milvus
from langchain_core.vectorstores import VectorStore
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from llmx import TextGenerator, TextGenerationConfig

from lida.datamodel import ChartExecutorResponse, Summary
from lida.utils import get_project_root

logger = logging.getLogger("lida")


class EmbeddingModel:
    """Factory class for embedding models"""

    @staticmethod
    def create(provider: str = "openai", **kwargs) -> Any:
        """Create an embedding model based on the provider

        Args:
            provider (str, optional): Provider name. Defaults to "openai".
            **kwargs: Additional keyword arguments for the provider.

        Returns:
            Any: Embedding model instance
        """
        if provider == "openai":
            return OpenAIEmbeddings(**kwargs)
        elif provider == "cohere":
            return CohereEmbeddings(**kwargs)
        elif provider == "huggingface":
            return HuggingFaceEmbeddings(**kwargs)
        elif provider == "google":
            return GoogleGenerativeAIEmbeddings(**kwargs)
        else:
            raise ValueError(f"Unsupported embedding model provider: {provider}")


class VectorDatabase:
    """Factory class for vector databases"""

    @staticmethod
    def create(
        embedding_model, provider: str = "faiss", persist_dir: str = None, **kwargs
    ) -> VectorStore:
        """Create a vector database based on the provider

        Args:
            embedding_model: Embedding model to use
            provider (str, optional): Provider name. Defaults to "faiss".
            persist_dir (str, optional): Directory to persist the database. Defaults to None.
            **kwargs: Additional keyword arguments for the provider.

        Returns:
            VectorStore: Vector store instance
        """
        if provider == "faiss":
            if persist_dir and os.path.exists(os.path.join(persist_dir, "index.faiss")):
                return FAISS.load_local(persist_dir, embedding_model)
            return FAISS.from_documents([], embedding_model)
        elif provider == "milvus":
            return Milvus(embedding_function=embedding_model, **kwargs)
        else:
            raise ValueError(f"Unsupported vector database provider: {provider}")


class MultimodalRAGPipeline:
    """Multimodal RAG Pipeline for LIDA"""

    def __init__(
        self,
        embedding_provider: str = "openai",
        embedding_kwargs: Dict = None,
        vector_db_provider: str = "faiss",
        vector_db_kwargs: Dict = None,
        llm_provider: str = "openai",
        llm_kwargs: Dict = None,
    ):
        """Initialize the multimodal RAG pipeline

        Args:
            embedding_provider (str, optional): Embedding model provider. Defaults to "openai".
            embedding_kwargs (Dict, optional): Embedding model kwargs. Defaults to None.
            vector_db_provider (str, optional): Vector database provider. Defaults to "faiss".
            vector_db_kwargs (Dict, optional): Vector database kwargs. Defaults to None.
            llm_provider (str, optional): LLM provider. Defaults to "openai".
            llm_kwargs (Dict, optional): LLM kwargs. Defaults to None.
        """
        self.embedding_provider = embedding_provider
        self.embedding_kwargs = embedding_kwargs or {}
        self.vector_db_provider = vector_db_provider
        self.vector_db_kwargs = vector_db_kwargs or {}
        self.llm_provider = llm_provider
        self.llm_kwargs = llm_kwargs or {}

        # Initialize embedding model
        self.embedding_model = EmbeddingModel.create(
            provider=embedding_provider, **self.embedding_kwargs
        )

        # Set up storage for raw content (images, text chunks, tables)
        self.byte_store = InMemoryStore()  # Can be replaced with Redis, Filesystem, etc.

        # Initialize vector database for storing embeddings
        persist_dir = os.path.join(get_project_root(), "vectorstore")
        os.makedirs(persist_dir, exist_ok=True)
        
        self.vector_db_kwargs["persist_dir"] = persist_dir
        self.vector_db = VectorDatabase.create(
            self.embedding_model,
            provider=vector_db_provider,
            **self.vector_db_kwargs
        )

        # Initialize MultiVector retriever
        self.retriever = MultiVectorRetriever(
            vectorstore=self.vector_db,
            docstore=self.byte_store,
            id_key="doc_id",
        )

        # Initialize LLM
        if llm_provider == "openai":
            self.llm = ChatOpenAI(**self.llm_kwargs)
        elif llm_provider == "google":
            self.llm = ChatGoogleGenerativeAI(**self.llm_kwargs)
        else:
            raise ValueError(f"Unsupported LLM provider: {llm_provider}")

        # Initialize text splitter for chunking documents
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=100
        )

    def _image_to_base64(self, image_data: bytes) -> str:
        """Convert image bytes to base64 string

        Args:
            image_data (bytes): Image bytes

        Returns:
            str: Base64 encoded image
        """
        return base64.b64encode(image_data).decode("utf-8")

    def _extract_chart_image(self, chart: ChartExecutorResponse) -> Optional[bytes]:
        """Extract chart image bytes from chart object

        Args:
            chart (ChartExecutorResponse): Chart response object

        Returns:
            Optional[bytes]: Image bytes or None
        """
        if chart.raster:
            # If raster is already available as base64, decode it
            return base64.b64decode(chart.raster)
        
        # For non-raster charts (e.g., vega-lite), we need to render them
        # This would require a secure execution environment
        # For now, we return None
        return None

    def analyze_chart(
        self, 
        chart: ChartExecutorResponse, 
        query: str = "Analyze this chart and provide insights", 
        k: int = 3
    ) -> str:
        """Analyze a chart using multimodal RAG

        Args:
            chart (ChartExecutorResponse): Chart to analyze
            query (str, optional): User query. Defaults to "Analyze this chart and provide insights".
            k (int, optional): Number of documents to retrieve. Defaults to 3.

        Returns:
            str: Analysis results
        """
        # Extract chart image
        image_bytes = self._extract_chart_image(chart)
        if not image_bytes:
            return "Unable to extract chart image for analysis"

        # Convert image to base64 for the multimodal LLM
        image_b64 = self._image_to_base64(image_bytes)
        
        # Define prompt for chart analysis
        chart_analysis_prompt = PromptTemplate(
            template="""
            Analyze the following chart and provide detailed insights:

            Chart Code:
            {chart_code}

            <image_base64>{image_b64}</image_base64>

            Based on the chart and the retrieved context, answer the following query:
            {query}

            Retrieved Context:
            {context}

            Your analysis should include:
            1. A description of what the chart is showing
            2. Key patterns, trends, or outliers visible in the visualization
            3. Insights that can be derived from combining the visual data with the retrieved context
            4. Potential actionable recommendations based on these insights

            Analysis:
            """,
            input_variables=["chart_code", "image_b64", "query", "context"],
        )

        # Set up the RAG chain
        rag_chain = (
            {
                "context": self.retriever.get_relevant_documents(query),
                "chart_code": lambda _: chart.code,
                "image_b64": lambda _: image_b64,
                "query": lambda _: query,
            }
            | RunnablePassthrough.assign(
                context=lambda x: "\n\n".join([doc.page_content for doc in x["context"]])
            )
            | chart_analysis_prompt
            | self.llm
            | StrOutputParser()
        )

        # Execute the chain
        return rag_chain.invoke({})

    def ingest_pdf(self, file_path: str) -> None:
        """Ingest a PDF document into the RAG system

        Args:
            file_path (str): Path to the PDF file
        """
        # Load PDF
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        
        # Split into chunks
        chunks = self.text_splitter.split_documents(documents)
        
        # Process each chunk and add to retriever
        for i, chunk in enumerate(chunks):
            # Create a unique ID for this chunk
            doc_id = f"pdf_{os.path.basename(file_path)}_{i}"
            
            # Add to byte store and vector store
            self.byte_store.mset([(doc_id, chunk)])
            
            # Create a summary document to be embedded
            summary_doc = Document(
                page_content=chunk.page_content,
                metadata={"doc_id": doc_id, "source": file_path, **chunk.metadata}
            )
            
            # Add to vector store
            self.vector_db.add_documents([summary_doc])

    def ingest_summary(self, summary: Summary) -> None:
        """Ingest a LIDA summary into the RAG system

        Args:
            summary (Summary): LIDA summary object
        """
        # Convert summary to a document
        summary_text = f"Dataset: {summary.name}\n"
        summary_text += f"Description: {summary.dataset_description}\n\n"
        summary_text += "Fields:\n"
        
        for field in summary.fields:
            summary_text += f"- {field.column}: {field.properties.dtype}\n"
            if hasattr(field.properties, "description") and field.properties.description:
                summary_text += f"  Description: {field.properties.description}\n"
            if hasattr(field.properties, "semantic_type") and field.properties.semantic_type:
                summary_text += f"  Semantic Type: {field.properties.semantic_type}\n"
            if hasattr(field.properties, "samples") and field.properties.samples:
                samples = ", ".join([str(s) for s in field.properties.samples[:3]])
                summary_text += f"  Samples: {samples}\n"
        
        # Create a unique ID for this summary
        doc_id = f"summary_{summary.name}"
        
        # Create a document to be embedded
        doc = Document(
            page_content=summary_text,
            metadata={"doc_id": doc_id, "source": "lida_summary", "name": summary.name}
        )
        
        # Add to byte store and vector store
        self.byte_store.mset([(doc_id, doc)])
        self.vector_db.add_documents([doc])

    def ingest_web_content(self, url: str, content: str) -> None:
        """Ingest web content into the RAG system

        Args:
            url (str): URL of the web content
            content (str): Content to ingest
        """
        # Split content into chunks
        texts = self.text_splitter.split_text(content)
        
        # Process each chunk
        for i, text in enumerate(texts):
            # Create a unique ID for this chunk
            doc_id = f"web_{url.replace('/', '_')}_{i}"
            
            # Create a document to be embedded
            doc = Document(
                page_content=text,
                metadata={"doc_id": doc_id, "source": url}
            )
            
            # Add to byte store and vector store
            self.byte_store.mset([(doc_id, doc)])
            self.vector_db.add_documents([doc])

    def update_knowledge_base(self, source_path: str = None) -> None:
        """Update the knowledge base by processing new or changed sources

        Args:
            source_path (str, optional): Path to source directory. Defaults to None.
        """
        if source_path is None:
            source_path = os.path.join(get_project_root(), "data")
        
        # Check if the source directory exists
        if not os.path.exists(source_path):
            logger.warning(f"Source directory {source_path} does not exist")
            return
        
        # Get list of PDF files in the source directory
        pdf_files = [
            os.path.join(source_path, f) 
            for f in os.listdir(source_path) 
            if f.lower().endswith(".pdf")
        ]
        
        # Process each PDF file
        for pdf_file in pdf_files:
            # Check if the file has been modified since last ingestion
            # For now, we'll just re-ingest all files
            # In a production system, you would check file modification times
            self.ingest_pdf(pdf_file)
        
        logger.info(f"Updated knowledge base with {len(pdf_files)} PDF files")


class Analyzer:
    """LIDA Analysis Component using Multimodal RAG"""

    def __init__(self, text_gen: TextGenerator = None):
        """Initialize the Analyzer component

        Args:
            text_gen (TextGenerator, optional): Text generator object. Defaults to None.
        """
        self.text_gen = text_gen
        
        # Default configuration
        default_config = {
            "embedding_provider": "openai",
            "embedding_kwargs": {"model": "text-embedding-3-small"},
            "vector_db_provider": "faiss",
            "vector_db_kwargs": {},
            "llm_provider": "openai",
            "llm_kwargs": {"model_name": "gpt-4o", "temperature": 0.2},
        }
        
        # Initialize the RAG pipeline
        self.rag_pipeline = MultimodalRAGPipeline(**default_config)

    def analyze(
        self,
        chart: ChartExecutorResponse,
        summary: Summary = None,
        query: str = "Analyze this chart and provide insights",
        textgen_config: TextGenerationConfig = TextGenerationConfig(),
        context_k: int = 3,
    ) -> str:
        """Analyze a chart using multimodal RAG

        Args:
            chart (ChartExecutorResponse): Chart to analyze
            summary (Summary, optional): Data summary. Defaults to None.
            query (str, optional): User query. Defaults to "Analyze this chart and provide insights".
            textgen_config (TextGenerationConfig, optional): Text generation config. Defaults to TextGenerationConfig().
            context_k (int, optional): Number of context documents to retrieve. Defaults to 3.

        Returns:
            str: Analysis results
        """
        # If summary is provided, ingest it
        if summary:
            self.rag_pipeline.ingest_summary(summary)
        
        # Analyze the chart using the RAG pipeline
        result = self.rag_pipeline.analyze_chart(chart, query, k=context_k)
        return result

    def ingest_document(self, file_path: str) -> None:
        """Ingest a document into the RAG system

        Args:
            file_path (str): Path to the document
        """
        if file_path.lower().endswith(".pdf"):
            self.rag_pipeline.ingest_pdf(file_path)
        else:
            logger.warning(f"Unsupported document type: {file_path}")

    def update_knowledge_base(self, source_path: str = None) -> None:
        """Update the knowledge base

        Args:
            source_path (str, optional): Path to source directory. Defaults to None.
        """
        self.rag_pipeline.update_knowledge_base(source_path)

    def ingest_web_content(self, url: str, content: str) -> None:
        """Ingest web content into the RAG system

        Args:
            url (str): URL of the web content
            content (str): Content to ingest
        """
        self.rag_pipeline.ingest_web_content(url, content) 