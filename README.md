# PDF Question Answering System

Hybrid PDF extraction + RAG-based question answering with semantic search and intelligent caching.

## Quick Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Add OpenAI API key to .env
echo "OPENAI_API_KEY=your_key_here" > .env
```

## Usage

### 1. Interactive Q&A (Recommended for Testing)
```bash
python files/interactive_qa.py
```
Enter questions interactively about PDFs in `data/` folder.

### 2. Programmatic Usage
```python
from files.Hybrid_pdf_qa_optimized import answer_pdf_question

answer = answer_pdf_question(
    question="What are the coverage limits?",
    pdfs_folder="./data"
)
print(answer)
```


### 3. Experimentation Framework
```bash
python files/experimentation_harness.py
```
Compares 4 configurations across 3 test cases with detailed metrics.

## Project Structure

```
├── data/                          # PDF files
├── files/
│   ├── Hybrid_pdf_qa_optimized.py # Main QA system
│   ├── experimentation_harness.py # Testing framework
│   └── interactive_qa.py          # Interactive CLI
├── .pdf_qa_cache/                 # Auto-generated cache
└── requirements.txt               # Dependencies
```

## Key Features

- **Hybrid Extraction**: PDFPlumber (tables) + Unstructured (document structure)
- **Smart Chunking**: Preserves tables, 2500 chars with 400 char overlap
- **Semantic Search**: OpenAI embeddings with domain-specific boosting
- **Persistent Caching**: Disk cache for chunks + embeddings (saves API costs)
- **Query-Adaptive**: Dynamic retrieval (20/35/60 chunks based on query type)
- **Specialized Prompts**: Different prompts for calculations, lists, and general queries

## Cache Benefits

- **First run**: Full processing + API calls
- **Subsequent runs**: ~3x faster, near-zero cost (cached embeddings)
- Cache invalidates automatically when PDFs change

