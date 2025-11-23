# 🖥️ Web Interface Guide

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Set API Key (Optional for Part C)
```bash
export OPENAI_API_KEY=your_key_here
```

### 3. Run the Interface
```bash
streamlit run app.py
```

Or use the helper script:
```bash
./run_interface.sh
```

## Features

The web interface provides:

### 📊 Overview Page
- Project summary
- Dataset information
- File structure
- Quick stats

### Part A: Email Tagging
- Run Part A with one click
- View output in real-time
- See classification results
- Access documentation

### Part B: Sentiment Analysis
- Run Part B evaluation
- Compare v1 vs v2 results
- View sentiment classifications
- Read evaluation report

### Part C: Mini-RAG
- Run RAG system
- View retrieved articles
- See generated answers
- Check confidence scores

### Run All
- Execute all parts sequentially
- Progress tracking
- Summary of results
- Combined output view

## Recording Tips

1. **Start Recording**: Begin screen recording before opening the interface
2. **Show Overview**: Start with the Overview page to show project structure
3. **Run Each Part**: Navigate to each part and click "Run"
4. **Show Results**: Expand results and outputs
5. **Documentation**: Show the documentation sections
6. **Final Summary**: Use "Run All" to show complete execution

## Interface URL

The interface will open at: **http://localhost:8501**

## Troubleshooting

- **Port already in use**: Change port with `streamlit run app.py --server.port 8502`
- **API Key not found**: Set it in environment or .env file
- **Module not found**: Run `pip install -r requirements.txt`

