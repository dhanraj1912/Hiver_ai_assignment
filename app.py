"""
Streamlit Web Interface for Hiver AI Assignment
Provides a user-friendly interface to run all three parts and view outputs.
"""

import streamlit as st
import subprocess
import sys
import os
import pandas as pd
from pathlib import Path
import json

# Page configuration
st.set_page_config(
    page_title="Hiver AI Assignment",
    page_icon="📧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better appearance
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .part-header {
        font-size: 1.8rem;
        font-weight: bold;
        color: #2c3e50;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 3px solid #3498db;
        padding-bottom: 0.5rem;
    }
    .success-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        margin: 1rem 0;
    }
    .info-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        color: #0c5460;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def run_script(script_name, description):
    """Run a Python script and capture output."""
    st.info(f"Running {description}...")
    
    try:
        result = subprocess.run(
            [sys.executable, script_name],
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        if result.returncode == 0:
            st.success(f"✅ {description} completed successfully!")
            return result.stdout, result.stderr, True
        else:
            st.error(f"❌ {description} encountered an error")
            return result.stdout, result.stderr, False
    except subprocess.TimeoutExpired:
        st.error(f"⏱️ {description} timed out after 5 minutes")
        return "", "Timeout error", False
    except Exception as e:
        st.error(f"❌ Error running {description}: {str(e)}")
        return "", str(e), False

def main():
    # Header
    st.markdown('<div class="main-header">📧 Hiver AI Intern Evaluation Assignment</div>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("🎯 Navigation")
        page = st.radio(
            "Select Part",
            ["Overview", "Part A: Email Tagging", "Part B: Sentiment Analysis", "Part C: Mini-RAG", "Run All"]
        )
        
        st.markdown("---")
        st.header("📋 Quick Info")
        st.info("""
        **Requirements:**
        - OpenAI API Key (for Parts A & B)
        - Python packages installed
        - Dataset CSV file
        
        **Note:** Part C can run without API key (uses open-source embeddings)
        """)
        
        # Check API key
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            st.success("✅ OpenAI API Key found")
        else:
            st.warning("⚠️ OpenAI API Key not set")
            st.caption("Set it in environment or .env file")
    
    # Overview Page
    if page == "Overview":
        st.markdown('<div class="part-header">📊 Project Overview</div>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("### Part A: Email Tagging")
            st.markdown("""
            - Customer-specific tagging
            - Customer isolation
            - Pattern-based improvements
            - Error analysis
            """)
        
        with col2:
            st.markdown("### Part B: Sentiment Analysis")
            st.markdown("""
            - Prompt v1 & v2 evaluation
            - Consistency testing
            - Systematic evaluation guide
            """)
        
        with col3:
            st.markdown("### Part C: Mini-RAG")
            st.markdown("""
            - Vector retrieval system
            - KB article answering
            - Confidence scoring
            """)
        
        st.markdown("---")
        
        # File structure
        st.markdown("### 📁 Project Files")
        files = [
            "dataset.csv - 60 customer support emails",
            "part_a_email_tagging.py - Part A implementation",
            "part_b_sentiment_analysis.py - Part B implementation",
            "part_c_rag_system.py - Part C implementation",
            "README.md - Main documentation",
            "README_PART_A.md - Part A documentation",
            "README_PART_C.md - Part C documentation",
            "part_b_report.md - Part B evaluation report",
            "requirements.txt - Dependencies"
        ]
        
        for file in files:
            st.markdown(f"- {file}")
        
        # Dataset info
        st.markdown("---")
        st.markdown("### 📊 Dataset Information")
        if Path("dataset.csv").exists():
            df = pd.read_csv("dataset.csv")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Emails", len(df))
            with col2:
                st.metric("Customers", df['customer_id'].nunique())
            with col3:
                st.metric("Unique Tags", df['tag'].nunique())
            
            st.dataframe(df.head(10), use_container_width=True)
        else:
            st.error("dataset.csv not found!")
    
    # Part A
    elif page == "Part A: Email Tagging":
        st.markdown('<div class="part-header">Part A: Email Tagging Mini-System</div>', unsafe_allow_html=True)
        
        st.markdown("""
        This system implements customer-specific email tagging with:
        - **Customer Isolation**: Tags from one customer never leak to another
        - **Pattern-based Classification**: Uses extracted patterns for accuracy
        - **Anti-pattern Guardrails**: Prevents common misclassifications
        - **Comprehensive Error Analysis**: Detailed breakdown of classification errors
        """)
        
        if st.button("🚀 Run Part A", type="primary", use_container_width=True):
            if not os.getenv("OPENAI_API_KEY"):
                st.error("⚠️ OpenAI API Key not set! Please set OPENAI_API_KEY environment variable.")
            else:
                stdout, stderr, success = run_script("part_a_email_tagging.py", "Part A: Email Tagging")
                
                if stdout:
                    st.markdown("### 📤 Output:")
                    st.code(stdout, language="text")
                
                if stderr and not success:
                    st.markdown("### ⚠️ Errors:")
                    st.code(stderr, language="text")
        
        # Show documentation
        with st.expander("📖 View Part A Documentation"):
            if Path("README_PART_A.md").exists():
                with open("README_PART_A.md", "r") as f:
                    st.markdown(f.read())
            else:
                st.error("README_PART_A.md not found")
    
    # Part B
    elif page == "Part B: Sentiment Analysis":
        st.markdown('<div class="part-header">Part B: Sentiment Analysis Prompt Evaluation</div>', unsafe_allow_html=True)
        
        st.markdown("""
        Evaluates sentiment analysis prompts for:
        - **Consistency**: Same input produces same output
        - **Accuracy**: Correct sentiment classification
        - **Debuggability**: Clear reasoning for classifications
        - **Systematic Evaluation**: Framework for prompt improvement
        """)
        
        if st.button("🚀 Run Part B", type="primary", use_container_width=True):
            if not os.getenv("OPENAI_API_KEY"):
                st.error("⚠️ OpenAI API Key not set! Please set OPENAI_API_KEY environment variable.")
            else:
                stdout, stderr, success = run_script("part_b_sentiment_analysis.py", "Part B: Sentiment Analysis")
                
                if stdout:
                    st.markdown("### 📤 Output:")
                    st.code(stdout, language="text")
                
                if stderr and not success:
                    st.markdown("### ⚠️ Errors:")
                    st.code(stderr, language="text")
                
                # Show results if available
                if Path("sentiment_results_v1.csv").exists():
                    st.markdown("### 📊 Results:")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**Prompt v1 Results:**")
                        df_v1 = pd.read_csv("sentiment_results_v1.csv")
                        st.dataframe(df_v1, use_container_width=True)
                    with col2:
                        if Path("sentiment_results_v2.csv").exists():
                            st.markdown("**Prompt v2 Results:**")
                            df_v2 = pd.read_csv("sentiment_results_v2.csv")
                            st.dataframe(df_v2, use_container_width=True)
        
        # Show report
        with st.expander("📖 View Part B Report"):
            if Path("part_b_report.md").exists():
                with open("part_b_report.md", "r") as f:
                    st.markdown(f.read())
            else:
                st.error("part_b_report.md not found")
    
    # Part C
    elif page == "Part C: Mini-RAG":
        st.markdown('<div class="part-header">Part C: Mini-RAG for Knowledge Base Answering</div>', unsafe_allow_html=True)
        
        st.markdown("""
        RAG system that:
        - **Retrieves** relevant KB articles using embeddings
        - **Generates** answers based on retrieved context
        - **Provides** confidence scores for answers
        - **Answers** two specific queries about Hiver
        """)
        
        if st.button("🚀 Run Part C", type="primary", use_container_width=True):
            stdout, stderr, success = run_script("part_c_rag_system.py", "Part C: Mini-RAG")
            
            if stdout:
                st.markdown("### 📤 Output:")
                st.code(stdout, language="text")
            
            if stderr and not success:
                st.markdown("### ⚠️ Errors:")
                st.code(stderr, language="text")
            
            # Show results if available
            if Path("rag_results.json").exists():
                st.markdown("### 📊 Results:")
                with open("rag_results.json", "r") as f:
                    results = json.load(f)
                
                for query_key in ["query1", "query2"]:
                    if query_key in results:
                        query_data = results[query_key]
                        st.markdown(f"#### Query: {query_data['query']}")
                        st.markdown(f"**Answer:** {query_data['answer']}")
                        st.markdown(f"**Confidence:** {query_data['confidence']:.2%}")
                        st.markdown(f"**Retrieved Articles:** {', '.join(query_data['retrieved_article_ids'])}")
                        st.markdown("---")
        
        # Show documentation
        with st.expander("📖 View Part C Documentation"):
            if Path("README_PART_C.md").exists():
                with open("README_PART_C.md", "r") as f:
                    st.markdown(f.read())
            else:
                st.error("README_PART_C.md not found")
    
    # Run All
    elif page == "Run All":
        st.markdown('<div class="part-header">🚀 Run All Parts</div>', unsafe_allow_html=True)
        
        st.warning("⚠️ This will run all three parts sequentially. Part A and B require OpenAI API key.")
        
        if st.button("▶️ Run All Parts", type="primary", use_container_width=True):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Part A
            status_text.text("Running Part A: Email Tagging...")
            progress_bar.progress(33)
            stdout_a, stderr_a, success_a = run_script("part_a_email_tagging.py", "Part A")
            
            # Part B
            status_text.text("Running Part B: Sentiment Analysis...")
            progress_bar.progress(66)
            stdout_b, stderr_b, success_b = run_script("part_b_sentiment_analysis.py", "Part B")
            
            # Part C
            status_text.text("Running Part C: Mini-RAG...")
            progress_bar.progress(100)
            stdout_c, stderr_c, success_c = run_script("part_c_rag_system.py", "Part C")
            
            # Summary
            status_text.text("✅ All parts completed!")
            st.markdown("### 📊 Summary")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                if success_a:
                    st.success("✅ Part A: Success")
                else:
                    st.error("❌ Part A: Failed")
            with col2:
                if success_b:
                    st.success("✅ Part B: Success")
                else:
                    st.error("❌ Part B: Failed")
            with col3:
                if success_c:
                    st.success("✅ Part C: Success")
                else:
                    st.error("❌ Part C: Failed")
            
            # Show all outputs
            with st.expander("📤 View All Outputs"):
                st.markdown("#### Part A Output:")
                st.code(stdout_a if stdout_a else "No output", language="text")
                st.markdown("#### Part B Output:")
                st.code(stdout_b if stdout_b else "No output", language="text")
                st.markdown("#### Part C Output:")
                st.code(stdout_c if stdout_c else "No output", language="text")

if __name__ == "__main__":
    main()

