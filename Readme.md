# Tiny-Universe Study Notes

![Tiny-Universe](https://img.shields.io/badge/Tiny-Universe-blue?style=flat-square) ![Study Notes](https://img.shields.io/badge/Study-Notes-green?style=flat-square) ![LLM](https://img.shields.io/badge/LLM-Learning-orange?style=flat-square)

This repository contains my study notes and implementations based on the [Datawhale Tiny-Universe project](https://github.com/datawhalechina/tiny-universe) - "A comprehensive guide to building LLM systems from scratch" (《大模型白盒子构建指南》).

## 📁 Repository Structure


### Task 1: Qwen Model Deep Dive 🔍
**Directory**: `Task1/`

- **Focus**: Understanding Qwen2 model architecture and internal mechanisms
- **Key Components**:
  - Model configuration and initialization
  - Decoder layer implementation
  - Attention mechanism (including GQA - Grouped Query Attention)
  - Position embeddings and RoPE
  - Forward pass walkthrough

### Task 2: TinyLLM - Pretraining from Scratch 🚀
**Directory**: `Task2/`

- **Focus**: Building and pretraining a Llama3-style model from scratch
- **Key Components**:
  - Model pretraining pipeline
  - Data preparation and tokenization
  - Training loop implementation
  - Model inference and text generation

### Task 3: TinyAgent - Building an AI Agent 🤖
**Directory**: `Task3/`

- **Focus**: Implementing a minimal Agent system using the ReAct paradigm
- **Key Components**:
  - ReAct (Reasoning + Acting) framework implementation
  - Tool integration (Google Search)
  - Agent planning and execution logic
  - System prompt engineering
- **Architecture**: Two-stage model calling for tool selection and response generation

### Task 4: TinyEval - LLM Evaluation Framework 📊
**Directory**: `Task4/`

- **Focus**: Building a comprehensive evaluation system for LLMs
- **Key Components**:
  - Multi-modal evaluation (generative, discriminative, choice-based)
  - Multiple metrics (F1, ROUGE, BLEU, Accuracy)
  - Custom dataset evaluation support
  - Two-stage evaluation pipeline (inference + evaluation)
- **Supported Tasks**: Question answering, text generation, classification



---

*This repository represents my personal learning journey through the fascinating world of Large Language Models.*

**Happy Learning! 🎉**
