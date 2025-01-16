# 🎓 Bachelor Thesis: Automated Extraction and Structuring of Relevant Information from Neuroscientific Documents Using Cosine Similarity and Large Language Model (LLM)

## 📑 Table of Contents
- [Introduction](#introduction)
- [System Design](#methodology)
- [Results](#results)
- [Tools Used](#tools-used)

## 📑 Introduction

## Overview

This repository consists of 4 main files:

1. **compare_similarities.ipynb**
This file contains similarity matrix calculation and comparisons of similarity matrices according to similarity function and embedding model, along with plotting.

2. **llm_rag_information_extraction.py**
This file contains text generation with LLM using RAG and exporting LLM result CSV.

3. results_evaluate.py**
This file contains post-processing of LLM results and pre-processing of Eilts' results. It also applies ranking to both results and calculates correlation coefficients for them.

4. **results_plotting.ipynb**
This file contains plotting LLM results.

**Knowledge Base:** data/pdf_documents 

![Status](https://img.shields.io/badge/Status-Completed-green)


---

## System Design

<details>
<summary>Click to view the system design diagram</summary>

![System Design](images/diagram.png)

</details>

---

## Some Results

### Removing Bibliography Part From Document

<details>
<summary>Click to view bibliography results</summary>

![Bibliography Result](images/bib-AA.png)

![Bibliography Result](images/bib-IEA.png)

![Bibliography Result](images/bib-MI.png)

</details>

---

### LLMs Comparison

<details>
<summary>Click to view LLM comparison results</summary>

![LLM Models Comparison for AA](images/llm-models-AA-S.png)

![LLM Models Comparison for IEA](images/llm-models-IEA-S.png)

![LLM Models Comparison for MI](images/llm-models-MI-S.png)

</details>
