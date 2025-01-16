# 🎓 Bachelor Thesis: Automated Extraction and Structuring of Relevant Information from Neuroscientific Documents Using Cosine Similarity and Large Language Model (LLM)

## 📑 Table of Contents
- [Introduction](#introduction)
- [System Design](#system-design)
- [File Overview](#file-overview)
- [Results](#results)
- [Tools Used](#tools-used)

## 📘 Introduction
This thesis utilizes cosine similarity and large language model (LLM) with Retrieval Augmented Generation (RAG) to extract information from neuroscientific documents containing information about three electroencephalography (EEG) topics motor imagery, auditory attention, internal- external attention. The aim is to automate structured knowledge extraction from these documents in a faster and more precise way,

## System Design

The following image illustrates the general workflow of the methods:

![System Design](images/diagram.png)

## File Overview

This repository consists of 4 main files:

1. **compare_similarities.ipynb**
This file contains similarity matrix calculation and comparisons of similarity matrices according to similarity function and embedding model, along with plotting.

2. **llm_rag_information_extraction.py**
This file contains text generation with LLM using RAG and exporting LLM result CSV.

3. results_evaluate.py**
This file contains post-processing of LLM results and pre-processing of Eilts' results. It also applies ranking to both results and calculates correlation coefficients for them.

4. **results_plotting.ipynb**
This file contains plotting LLM results.

**Knowledge Base:** 20 documents per topic in [pdf_documents](./data/pdf_documents) directory.

**Ground Truth:** Ranked results from Eilts' master thesis (Eilts, Hendrik. „Bridging the Gap: Explainable AI Insights Into EEGNet Classification and Its Alignment to Neural Correlates“. Advisor: Dr. Felix Putze; Reviewers: Dr. Felix Putze, Prof. Dr. Marvin Wright. MA thesis. Bremen, Germany: University of Bremen, Cognitive Systems Lab, Mar. 2024.) in [results](./data/results/ground_truth) directory.

---

## Results

### LLMs Comparison

![LLM Models Comparison for AA](images/llm-models-AA-S.png)

![LLM Models Comparison for IEA](images/llm-models-IEA-S.png)

![LLM Models Comparison for MI](images/llm-models-MI-S.png)

### Chunk Methods & Prompt Strategies

![Chunk Methods & Prompt Strategies for AA](images/chunk-method-AA-S.png)

![Chunk Methods & Prompt Strategies for IEA](images/chunk-method-IEA-S.png)

![Chunk Methods & Prompt Strategies for MI](images/chunk-method-MI-S.png)

### Chunking Combinations

![Chunk Combinations for AA](images/chunk-comb-AA.png)

![Chunk Combinations for IEA](images/chunk-comb-IEA-S.png)

![Chunk Combinations for MI](images/chunk-comb-MI-S.png)

### Removing Bibliography Part From Document

![Bibliography Result](images/bib-AA.png)

![Bibliography Result](images/bib-IEA.png)

![Bibliography Result](images/bib-MI.png)

### Chunk & Overlap Size

![Chunk & Overlap Size Result](images/chunk-size-AA-S.png)

![Chunk & Overlap Size Result](images/chunk-size-IEA-S.png)

![Chunk & Overlap Size Result](images/chunk-size-MI-S.png)

### Repetition Penalty

![Repetition Penalty for AA](images/rep-AA.png)

![Repetition Penalty for IEA](images/rep-IE.png)

![Repetition Penalty for MI](images/rep-MI.png)

## Tools Used
# Tools and Frameworks Used

## 1. Programming Language
- **Python**: The primary language used for implementing the workflow.

## 2. Libraries and Frameworks

### Machine Learning & NLP
- **torch**: A deep learning framework for handling models like Meta-Llama.
- **transformers**: Hugging Face library for model loading and text generation pipelines.
  - `AutoTokenizer` and `AutoModelForCausalLM`: Used for loading and managing the LLM (e.g., Meta-Llama-3.1-70B-AQLM-PV).
  - `pipeline`: A high-level abstraction for running inference tasks like text generation.
- **sentence-transformers**: For semantic embeddings like `all-MiniLM-L6-v2`.
  - `HuggingFaceEmbeddings` and `SentenceTransformer`: Used for embedding documents and sentences for similarity calculations.
- **langchain_community**:
  - `HuggingFacePipeline`: Integration for Hugging Face pipelines.
  - `PyPDFLoader`: PDF document loader for text extraction.
  - `RecursiveCharacterTextSplitter`: Splits text into chunks for processing.
  - `LLMChain`: Manages chain logic for large language models with prompt templates.
  - `PromptTemplate`: Templates for generating summaries, extracting relevant bands, formatting, and applying chain-of-thought prompting.
- **FAISS**: Vector database for similarity-based retrieval and indexing.

### Statistical Analysis
- **scipy.stats**:
  - `spearmanr`: For Spearman's rank correlation.
  - `kendalltau`: For Kendall's Tau correlation.
- **numpy**: For numerical computations, vector operations, and cosine similarity.
- **pandas**: For data manipulation, analysis, and saving results in CSV format.

### Data Handling
- **joblib**: For loading serialized objects.
- **os** and **pathlib**: For file handling and path operations.

### Text Processing
- **re**: Regular expressions for cleaning and preprocessing data.
- **nltk**: For sentence tokenization and semantic chunking.
- **spaCy**: For semantic chunking, lemmatization, stopword removal, and tokenization.

### Logging
- **logging**: Captures and debugs processing errors, storing logs in a file (`faiss_debug_dev.log`).

## 3. Models and Data

### Pretrained Models
- **Meta-Llama-3.1-70B-AQLM-PV**: Used for text generation and LLM-based tasks.
- **sentence-transformers/all-MiniLM-L6-v2**: Used for embedding generation.

### Datasets
- **Hendrik's XAI Results**: Preprocessed data for correlation and comparison.
- **Neuroscientific PDF Documents**: From directories such as Motor Imagery, Auditory Attention, and Internal/External Attention.

## 4. Semantic Chunking

### Techniques
- **RecursiveCharacterTextSplitter**: Used for chunking text into smaller pieces.
- **Semantic Chunking**:
  - **NLTK**: For sentence tokenization and similarity-based chunking.
  - **spaCy**: For fallback semantic chunking and lemmatization.

## 5. Preprocessing
- **Regex**: For removing unwanted sections like bibliographies.
- **spaCy**: For advanced preprocessing like lemmatization and stopword removal.

## 6. Knowledge Injection
- Domain-specific mappings of brain regions and electrode locations.
- Structured knowledge injected into prompts for enhanced context.

## 7. Storage and Retrieval
- **FAISS**: For vector-based document retrieval and indexing.

## 8. Integrated Methodologies

### Semantic Processing
- Domain-based, recursive, and semantic chunking approaches for effective document segmentation.

### Retrieval-Augmented Generation (RAG)
- Combining vector-based retrieval with LLMs for enhanced document querying and knowledge extraction.

### Chain-of-Thought Prompting
- Logical reasoning applied within LLM prompts for structured information extraction.

### Knowledge Augmentation
- Leveraging domain knowledge (e.g., brain regions, electrode locations) for precision in results.

### Correlation Analysis
- **Spearman’s Rank Correlation**: To compare ranked results.
- **Kendall’s Tau Correlation**: To validate rankings.

## 9. Filesystem Operations
- **Pathlib**: For directory and file handling for PDFs.
- **os**: For managing vector store directories and saving outputs.

## 10. Numerical Operations
- **NumPy**: For vector operations and cosine similarity calculations.

