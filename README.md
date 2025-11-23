# Hiver AI Intern Evaluation Assignment

This repository contains implementations for three parts of the Hiver AI Intern evaluation assignment:

- **Part A**: Email Tagging Mini-System
- **Part B**: Sentiment Analysis Prompt Evaluation  
- **Part C**: Mini-RAG for Knowledge Base Answering

## Project Structure

```
.
├── dataset.csv                    # Email dataset with customer support emails
├── requirements.txt               # Python dependencies
├── part_a_email_tagging.py       # Part A implementation
├── README_PART_A.md              # Part A documentation
├── part_b_sentiment_analysis.py  # Part B implementation
├── part_b_report.md              # Part B evaluation report
├── part_c_rag_system.py          # Part C implementation
├── README_PART_C.md              # Part C documentation
├── app.py                        # Streamlit web interface
├── run_interface.sh             # Helper script to run interface
├── INTERFACE_GUIDE.md            # Web interface guide
├── kb_articles/                  # Knowledge base articles (created by Part C)
└── README.md                     # This file
```

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Up OpenAI API Key

Create a `.env` file in the project root:

```bash
OPENAI_API_KEY=your_openai_api_key_here
```

Or set it as an environment variable:

```bash
export OPENAI_API_KEY=your_openai_api_key_here
```

**Note**: Part C can run without OpenAI API key (uses open-source embeddings), but Part A and Part B require it.

## Running the Projects

### 🖥️ Web Interface (Recommended for Recording)

For a user-friendly interface to run all parts and view outputs:

```bash
# Install dependencies (includes streamlit)
pip install -r requirements.txt

# Run the web interface
streamlit run app.py
```

Or use the helper script:
```bash
./run_interface.sh
```

The interface will open in your browser at `http://localhost:8501` and provides:
- One-click execution of each part
- Real-time output display
- Results visualization
- Documentation viewer
- "Run All" option for complete execution

**Perfect for screen recording and demonstrations!**

See `INTERFACE_GUIDE.md` for detailed instructions.

### Command Line Execution

### Part A: Email Tagging Mini-System

```bash
python part_a_email_tagging.py
```

**What it does:**
- Loads email dataset
- Builds customer-specific tag vocabularies (ensures customer isolation)
- Classifies emails using LLM prompt-based classifier
- Evaluates accuracy and performs error analysis
- Shows patterns and anti-patterns for accuracy improvement

**Key Features:**
- ✅ Customer isolation (tags from one customer don't leak to another)
- ✅ Pattern-based accuracy improvements
- ✅ Anti-pattern guardrails
- ✅ Comprehensive error analysis

**Documentation**: See `README_PART_A.md` for detailed approach and architecture.

### Part B: Sentiment Analysis Prompt Evaluation

```bash
python part_b_sentiment_analysis.py
```

**What it does:**
- Tests two versions of sentiment analysis prompts (v1 and v2)
- Evaluates on 10 sample emails
- Compares consistency and accuracy between versions
- Generates evaluation results

**Outputs:**
- `sentiment_results_v1.csv` - Results from prompt v1
- `sentiment_results_v2.csv` - Results from prompt v2
- `part_b_report.md` - Detailed evaluation report

**Documentation**: See `part_b_report.md` for:
- What failed in v1
- What was improved in v2
- How to evaluate prompts systematically

### Part C: Mini-RAG for Knowledge Base Answering

```bash
python part_c_rag_system.py
```

**What it does:**
- Creates sample KB articles (if they don't exist)
- Builds vector index using embeddings
- Answers two queries:
  1. "How do I configure automations in Hiver?"
  2. "Why is CSAT not appearing?"
- Shows retrieved articles, generated answers, and confidence scores

**Outputs:**
- `rag_results.json` - Query results with answers and confidence scores
- `kb_articles/` - Knowledge base articles folder

**Documentation**: See `README_PART_C.md` for:
- System architecture
- 5 ways to improve retrieval
- Failure case analysis and debugging steps

## Dataset

The `dataset.csv` file contains 60 customer support emails with:
- `email_id`: Unique identifier
- `customer_id`: Customer identifier (CUST_A through CUST_F)
- `subject`: Email subject line
- `body`: Email body text
- `tag`: Ground truth tag (for Part A evaluation)

## Key Design Decisions

### Part A: Customer Isolation
- Each customer has separate tag vocabulary
- Patterns and anti-patterns learned per customer
- Prompts explicitly restrict to customer-specific tags only

### Part B: Prompt Evaluation
- Systematic comparison of prompt versions
- Focus on consistency and debuggability
- Manual evaluation framework for quality assessment

### Part C: RAG System
- Open-source embeddings (sentence-transformers) as default
- ChromaDB for vector storage
- Hybrid confidence scoring (LLM + retrieval quality)

## Requirements

- Python 3.8+
- OpenAI API key (for Parts A & B, optional for Part C)
- See `requirements.txt` for full dependency list

## Evaluation Criteria Addressed

### Technical
- ✅ Clean thinking and architecture
- ✅ Quality prompts and model selection
- ✅ Understanding of customer segmentation (Part A)
- ✅ Basic understanding of embeddings + retrieval (Part C)
- ✅ Error analysis depth

### Engineering
- ✅ Readable, well-structured code
- ✅ Reproducibility (deterministic where possible)
- ✅ Simple architecture (no over-engineering)

### Communication
- ✅ Crisp documentation (README files)
- ✅ Clear explanations of approach
- ✅ Error analysis and improvements documented

## Notes

- Part C uses open-source embeddings by default (no API key needed)
- All parts are designed to be runnable end-to-end
- Error handling and fallbacks included
- Results are saved to files for review

## Contact

For questions or issues, please refer to the individual README files for each part.

