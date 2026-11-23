# L-Distill-RAG: Indian Legal Precedent & Statute Retrieval

A multi-stage retrieval and reranking framework designed for Indian legal information retrieval, targeting the IL-PCSR benchmark.

## Overview

This project implements a multi-stage retrieval architecture combining dense retrieval, sparse keyword matching, learning-to-rank, citation-graph reranking, and cross-task conditioning to improve the discovery of relevant Indian legal precedents and statutes.

## Architecture

1. **Dense Retrieval:** InLegalBERT embeddings with FAISS indexing.
2. **Sparse Retrieval:** BM25-based lexical retrieval.
3. **Learning to Rank:** LightGBM LambdaRank using a 17-feature ranking framework.
4. **Graph-Based Reranking:** DiffGraphRAG-based citation relationship modeling.
5. **Cross-Task Conditioning:** PCR-to-LSR Stage-2 conditioning.

## Tech Stack

Python | PyTorch | InLegalBERT | FAISS | BM25 | LightGBM | DiffGraphRAG | Llama | QLoRA

## Repository Structure

- `IL-PCSR-Retrieval-Pipeline.ipynb` — End-to-end Kaggle-ready implementation.

## Running the Project

1. Open the notebook in Kaggle.
2. Enable GPU acceleration if required.
3. Configure Hugging Face credentials through Kaggle Secrets if needed.
4. Run the notebook cells sequentially.
5. Review the generated retrieval metrics and evaluation outputs.

## Evaluation

Performance is evaluated on the IL-PCSR benchmark using PCR and LSR retrieval metrics, including F1@10.

See the notebook for implementation details and experimental results.

## Reproducibility

Dataset sources, dependencies, evaluation methodology, and model configurations should be documented alongside the implementation.
