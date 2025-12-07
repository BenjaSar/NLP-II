![header](doc/imgs/LogoHeader.png)

# NLP II - AI Agents & Language Models

**Author:** FS  
**Repository:** [NLP-II](https://github.com/BenjaSar/NLP-II)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Technology Stack](#technology-stack)
- [Assignments](#assignments)
  - [Assignment 1: GPT-3.5 Document QA Chatbot](#assignment-1-gpt-35-document-qa-chatbot)
  - [Assignment 2: CV Evaluator with GPT-4o & LangChain](#assignment-2-cv-evaluator-with-gpt-4o--langchain)
  - [Assignment 3: DeepSeek R1 Reasoning Agent](#assignment-3-deepseek-r1-reasoning-agent)
  - [Bonus: TinyGPT Implementation](#bonus-tinygpt-implementation)
- [Installation & Setup](#installation--setup)
- [Usage](#usage)
- [License](#license)

---

## 🎯 Overview

This repository contains a comprehensive collection of AI projects demonstrating advanced Natural Language Processing (NLP) techniques and autonomous AI agent development. The projects showcase integration with state-of-the-art language models, vector databases, and orchestration frameworks for building intelligent systems.

---

## 📁 Project Structure

```
Ejercicios/
├── tp2/                              # Assignment 2 & Bonus Projects
│   ├── chat_app.py                  # Streamlit chat application
│   ├── prompt.md                    # Prompt templates
│   ├── ai_engineer_cv.pdf           # Sample CV for evaluation
│   ├── utils/
│   │   └── streamlit.py             # Streamlit utilities
│   └── videos/                      # Demo videos
├── tinyGPT/                          # Bonus: Small GPT Implementation
│   ├── TinyGPT.ipynb                # Training notebook
│   ├── trainer.py                   # Training script
│   └── checkpoints/
│       └── checkpoint_final.pt      # Pre-trained model weights
├── text/                             # Testing & Utilities
│   ├── test_document_processor.py   # Document processing tests
│   └── test_qa_chain.py             # QA chain tests
├── doc/                              # Documentation & Assets
│   └── imgs/                        # Logo and header images
├── requirements.txt                  # Python dependencies
├── .env.example                      # Environment variables template
├── README.md                         # This file
└── AI_Agent*.ipynb                  # Jupyter notebooks for agents
```

---

## 🛠️ Technology Stack

- **Language Models:** OpenAI GPT-3.5, GPT-4o, DeepSeek-V2-R1
- **Frameworks:** LangChain, LangGraph, Streamlit
- **Vector Databases:** Pinecone, FAISS
- **Development:** Python, Jupyter Notebooks
- **Dependencies:** See `requirements.txt`

---

## 📚 Assignments

### Assignment 1: GPT-3.5 Document QA Chatbot

**Overview:** An intelligent chatbot leveraging OpenAI's GPT-3.5-turbo and semantic search to answer questions based on document content.

**Features:**
- ✅ Interactive Streamlit chat interface
- 🔍 Document-based question answering with semantic search
- 🤖 GPT-3.5-turbo LLM integration
- 🧠 Contextual retrieval using vector stores (Pinecone/FAISS)
- 📚 Source document referencing
- 💾 Persistent conversation history within session

**Demo:** See `Streamlit — chatbot.mp4` in the repository

---

### Assignment 2: CV Evaluator with GPT-4o & LangChain

**Overview:** A sophisticated multi-agent system for evaluating CVs and providing AI-powered recommendations using Retrieval-Augmented Generation (RAG).

**Features:**
- 🤖 Multiple autonomous AI agents with specialized roles
- 🔄 Retrieval-Augmented Generation (RAG) pipelines
- 📊 AgentState structured management for task coordination
- 🔗 LangGraph workflow orchestration with multi-step reasoning
- 🎯 Modular architecture: easy integration of custom planning, retrieval, and generation modules

**Architecture:**
```
Plan → Retrieve → Generate → Evaluate
```

**Demo:** See `AI_Agentv2.mp4` in the repository

---

### Assignment 3: DeepSeek R1 Reasoning Agent

**Overview:** An advanced AI agent powered by DeepSeek-V2-R1, designed for complex reasoning, planning, and recommendation generation.

**Capabilities:**
- 🔹 **Advanced Reasoning:** Step-by-step logical problem-solving
- 🔹 **Plan Generation:** Breaks complex tasks into actionable steps
- 🔹 **Self-Correction:** Reviews and improves answers iteratively
- 🔹 **Dynamic Calculations:** Performs numerical computations within reasoning
- 🔹 **LangChain Integration:** Supports chained workflows and memory management

**Features:**
- Structured multi-step reasoning
- Real-time calculation and verification
- Recommendation generation with confidence scores

**Demo:** See `AI_agent_Deepseek_reasoner.mp4` in the repository

---

### Bonus: TinyGPT Implementation

**Overview:** A lightweight GPT implementation demonstrating transformer architecture and training techniques.

**Contents:**
- `TinyGPT.ipynb` - Complete training pipeline notebook
- `trainer.py` - Standalone training script
- `checkpoint_final.pt` - Pre-trained model checkpoint

---

## 📦 Installation & Setup

### Prerequisites
- Python 3.8+
- pip or conda package manager
- Virtual environment (recommended)

### Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/BenjaSar/NLP-II.git
   cd Ejercicios
   ```

2. **Create and activate virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**
   ```bash
   cp .env.example .env
   ```
   
   Edit `.env` and add your API keys:
   ```env
   OPENAI_API_KEY=your_openai_key_here
   DEEPSEEK_API_KEY=your_deepseek_key_here
   PINECONE_API_KEY=your_pinecone_key_here
   PINECONE_ENV=your_pinecone_environment
   ```

---

## 🚀 Usage

### Running the Streamlit Chat Application
```bash
cd tp2
streamlit run chat_app.py
```

### Running Jupyter Notebooks
```bash
# Assignment 2: AI Agent with CV Evaluation
jupyter notebook AI_Agentv2.ipynb

# Assignment 3: DeepSeek Reasoning Agent
jupyter notebook AI_agent_Deepseek_reasoner.ipynb

# TinyGPT Training
jupyter notebook tinyGPT/TinyGPT.ipynb
```

### Running Tests
```bash
python text/test_document_processor.py
python text/test_qa_chain.py
```

---

## 📝 License

This project is licensed under the MIT License. See LICENSE file for details.

---

![footer](doc/imgs/LogoFooter.png)
