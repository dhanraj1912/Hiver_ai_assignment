# Part A: Email Tagging Mini-System - README

## Overview
This system implements a customer-specific email tagging classifier that ensures **customer isolation** - tags from one customer never leak into another customer's model.

## Approach

### 1. Customer Isolation Architecture
- **Separate Tag Vocabularies**: Each customer has their own set of tags that are learned only from their emails
- **Customer-Specific Context**: During prediction, only tags belonging to that customer are considered
- **Isolated Pattern Learning**: Patterns and anti-patterns are extracted per customer, not globally

### 2. Baseline Classifier
- **LLM Prompt-Based**: Uses GPT-4o-mini for classification
- **Structured Prompts**: Each prompt includes:
  - Customer-specific available tags only
  - Patterns extracted from that customer's historical data
  - Anti-patterns (guardrails) to prevent misclassification
  - Clear instructions to only use tags from the provided list

### 3. Model/Prompt Design

The classification prompt structure:
```
1. List of available tags (customer-specific)
2. Email subject and body
3. Patterns: Common keywords/phrases for each tag
4. Anti-patterns: Words that might mislead the model
5. JSON response format with tag, confidence, and reasoning
```

**Key Features:**
- Temperature: 0.1 (for consistency)
- JSON response format (structured output)
- Explicit customer ID in prompt to reinforce isolation

### 4. How Customer Isolation is Ensured

#### During Training/Setup:
1. **Separate Vocabulary Building**: 
   - For each customer, extract unique tags only from their emails
   - Store in `customer_tags[customer_id]` dictionary
   - No cross-customer tag sharing

2. **Pattern Extraction Per Customer**:
   - Patterns are learned from each customer's emails only
   - `customer_patterns[customer_id][tag]` contains patterns specific to that customer

#### During Prediction:
1. **Tag Filtering**: 
   - When classifying an email, retrieve tags only for that customer
   - `available_tags = list(self.customer_tags[customer_id])`
   - Other customers' tags are never included in the prompt

2. **Prompt Enforcement**:
   - Prompt explicitly states: "You can ONLY use tags from the list above"
   - Includes customer ID in prompt for additional context
   - Model cannot access tags from other customers

3. **Validation**:
   - System validates predicted tag is in customer's tag set
   - Fallback to first available tag if invalid prediction (shouldn't happen)

### 5. Patterns and Anti-Patterns

#### Patterns (Improving Accuracy):
- **Keyword Extraction**: For each tag, extract top 10 most frequent keywords from emails with that tag
- **Usage in Prompt**: Include patterns to guide the model toward correct classification
- **Example**: Tag "billing_error" might have patterns: ["charged", "invoice", "billing", "payment"]

#### Anti-Patterns (Guardrails):
- **Misleading Word Detection**: Identify words that appear in a tag's emails but also frequently in other tags
- **Purpose**: Prevent the model from over-relying on ambiguous words
- **Example**: Word "issue" appears in many tags - anti-pattern warns model to look for more specific context
- **Usage**: Included in prompt as guardrails: "Be cautious of words like X as they may appear in other contexts"

### 6. Error Analysis

The system performs comprehensive error analysis:

1. **Accuracy Metrics**:
   - Overall accuracy
   - Per-customer accuracy
   - Confusion matrix

2. **Error Categorization**:
   - Group errors by customer
   - Identify common misclassifications (confusion pairs)
   - Analyze confidence scores of errors

3. **Pattern Analysis**:
   - Review which patterns failed
   - Identify missing patterns
   - Update anti-patterns based on errors

## Usage

```bash
# Install dependencies
pip install -r requirements.txt

# Set OpenAI API key
export OPENAI_API_KEY=your_key_here

# Run the system
python part_a_email_tagging.py
```

## Output

The system outputs:
- Customer-specific tag vocabularies
- Classification accuracy
- Error analysis with common confusions
- Sample errors with reasoning

## 3 Major Improvement Ideas for Production

### 1. **Fine-Tuned Customer-Specific Models**
   - Instead of prompt-based classification, fine-tune small models per customer
   - Benefits: Better accuracy, lower latency, lower cost
   - Implementation: Use customer's historical data to fine-tune a base model
   - Trade-off: More complex deployment, but better performance

### 2. **Active Learning and Feedback Loop**
   - Collect user corrections and use them to improve patterns
   - Implement confidence-based sampling for human review
   - Continuously update patterns and anti-patterns
   - Benefits: System improves over time, adapts to customer-specific language

### 3. **Hybrid Approach: Rule-Based + LLM**
   - Use rule-based classifier for high-confidence cases (fast, cheap)
   - Fall back to LLM for ambiguous cases
   - Benefits: Lower cost, faster responses, maintainable rules
   - Implementation: Build rule engine using patterns, use LLM as fallback

## Architecture Diagram

```
┌─────────────────┐
│   Dataset CSV   │
└────────┬────────┘
         │
         ▼
┌─────────────────────────┐
│  Build Customer         │
│  Vocabularies           │
│  - Extract tags         │
│  - Extract patterns     │
│  - Extract anti-patterns│
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│  For each email:         │
│  1. Get customer tags    │
│  2. Build prompt         │
│  3. Call LLM             │
│  4. Validate tag         │
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│  Evaluate & Analyze      │
│  - Calculate accuracy    │
│  - Error analysis        │
│  - Confusion matrix      │
└─────────────────────────┘
```

## Key Design Decisions

1. **Why LLM-based?**: Flexible, handles diverse email content, easy to update prompts
2. **Why customer isolation?**: Privacy, prevents tag leakage, customer-specific terminology
3. **Why patterns + anti-patterns?**: Improves accuracy while preventing common mistakes
4. **Why JSON output?**: Structured, parseable, includes reasoning for debugging

## Limitations and Future Work

- Current pattern extraction is simple (keyword frequency) - could use more sophisticated NLP
- No learning from errors - could implement online learning
- Single model for all customers - could optimize per customer
- No handling of new tags - would need retraining/re-indexing

