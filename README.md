# DocQuery AI Chatbot

**DocQuery AI** is an intelligent chatbot designed to answer questions based on the content of a PDF document. If the chatbot cannot find an answer, it provides a fallback response, ensuring a seamless user experience. The chatbot leverages advanced document processing, query handling, and retrieval techniques to generate accurate responses.

---

## Features

- **PDF Understanding**: Extracts knowledge from PDFs and structures the information for efficient querying.
- **Intelligent Query Processing**: Handles complex user queries, including multi-part questions.
- **Accurate Responses**: Retrieves precise answers from the document.
- **Fallback Mechanism**: Responds with a default message when the answer is not found.
- **User-Friendly Interface**: Provides an intuitive interaction platform for users.

---

## Approach

The solution consists of three main components:

### 1. Document Processing

#### Part A: Conceptual Short Descriptions (CSD)
1. Treat the document as the knowledge base.
2. Divide the document into batches of 6 pages, creating parent documents.
3. For each parent document:
    - Generate conceptual short descriptions (CSD) using an LLM.
    - Combine all CSDs with pointers to their parent document.

#### Part B: Chunking for Semantic Retrieval
1. Split each parent document into smaller chunks.
2. Store these chunks for semantic retrieval during queries.

---

### 2. Query Processing

1. **Query Correction**:
   - Correct grammatical errors and refine the query for meaningful context.
   - Split multi-part queries into sub-queries.
2. **Retrieval Query Generation**:
   - Generate retrieval queries using an LLM.

---

### 3. Answering the Query

1. Process the user query through the Query Processing module.
2. Retrieve relevant parent documents and chunks:
   - Identify the top `k` parent documents.
   - Retrieve the top `m` chunks from each parent.
3. Combine the retrieved chunks and user query as context for the LLM.
4. Generate the final response using the LLM.

---

## Tools and Frameworks

### Frameworks
- **LangChain**: Workflow orchestration.
- **Ollama**: Backend for LLMs.
- **Gradio**: User interface.
- **HuggingFace**: Access to pre-trained models.

### LLMs
- **Local LLM**: `qwen2.5:7b`
- **Cloud LLM**: `Gemini-1.5-flash-002`

### Embedding Model
- **nomic-embed-text-v1.5**: Efficient local embedding model.

### Database
- **Faiss**: Scalable vector database for retrieval.

---

## Database Structure

1. **Conceptual Short Descriptions (CSDs)**:
   - Metadata: `parent_id`, `isparent=True`
2. **Chunks**:
   - Metadata: `parent_id`, `isparent=False`

Steps:
1. Assign `parent_id` to each parent document.
2. Add CSDs to the database with metadata.
3. Split parent documents into chunks and store with metadata.

---

## Prompts

1. **Generating CSDs**:
   - "For the given chunk of a document, provide concise one-line descriptions including entities, dates, and numbers."
2. **Query Correction**:
   - "Create queries that retrieve the correct context. Split multi-part questions. Return 'None' for general or unclear queries."
3. **Answering Queries**:
   - "Provide answers based on the context. If the context is missing or irrelevant, respond: *Sorry, I didn’t understand your question. Do you want to connect with a live agent?*"

---

## Optimizations

1. **Chat Memory Optimization**:
   - Store only user queries and model responses to reduce token usage.
2. **Contextual Query Handling**:
   - Automatically correct and refine follow-up queries.
3. **Prompt Compression**:
   - Reduce input token size by 50% while maintaining context.

---

## Hyperparameters

- **Parent Document Size**: ~6 pages (6 × 1800 characters).
- **Chunk Size**: ~900 characters.
- **Top Parents Retrieved**: 12 (deduplicated, keep top 2).
- **Chunks per Parent**: 3.
- **Final Context Size**: Combine 2 × 3 chunks with conceptual queries.

---

## Prerequisites

- Python 3.10 or higher
- Conda (recommended) or virtualenv
- poppler (required for PDF processing)
- Google Gemini API key

---

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/Rishikesh24062003/chatbot.git
cd chatbot
```

### 2. Install system dependencies

**macOS:**
```bash
brew install poppler
```

**Ubuntu/Debian:**
```bash
sudo apt-get install poppler-utils
```

**Windows:**
Download and install poppler from: https://github.com/oschwartz10612/poppler-windows/releases/

### 3. Create a Python environment

**Using Conda (recommended):**
```bash
conda create -n docquery_env python=3.10 -y
conda activate docquery_env
```

**Using venv:**
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 4. Install Python dependencies
```bash
pip install -r requirements.txt
```

### 5. Download NLTK data
```bash
python -c "import nltk; nltk.download('punkt_tab'); nltk.download('punkt'); nltk.download('averaged_perceptron_tagger'); nltk.download('averaged_perceptron_tagger_eng')"
```

### 6. Set up environment variables

Create a `.env` file in the project root or export the variable:

**Using .env file:**
```bash
echo "GOOGLE_API_KEY=your_api_key_here" > .env
```

**Or export directly:**
```bash
export GOOGLE_API_KEY="your_api_key_here"
```

To get a Google Gemini API key, visit: https://makersuite.google.com/app/apikey

### 7. Run the application
```bash
python app.py
```

The application will start on `http://127.0.0.1:7860`

---

## Usage

1. Upload a PDF document to the interface.
2. Ask questions based on the content of the document.
3. Receive accurate responses or fallback messages if the information is unavailable.

---

## Sample Interaction

- **User**: "When will admissions for BBA begin?"
- **Chatbot**: "Admissions for the Bachelor in Business Administration (BBA) program at NMIMS are open year-round, divided into two primary cycles..."

---

## Contribution

We welcome contributions to improve **DocQuery AI**! Please create a pull request with your proposed changes.

---

## License

This project is licensed under the MIT License. See the LICENSE file for details.
