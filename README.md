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
