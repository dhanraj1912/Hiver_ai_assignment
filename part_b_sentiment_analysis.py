"""
Part B: Sentiment Analysis Prompt Evaluation
Evaluates sentiment analysis prompts for consistency and debuggability.
"""

import pandas as pd
import json
import os
from typing import Dict, List, Tuple
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

class SentimentAnalyzer:
    """
    Sentiment analysis system that produces:
    - sentiment: positive/negative/neutral
    - confidence score
    - reasoning (hidden from end-user)
    """
    
    def __init__(self, api_key: str = None):
        """Initialize the analyzer with OpenAI client."""
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key required. Set OPENAI_API_KEY environment variable.")
        self.client = OpenAI(api_key=self.api_key)
    
    def load_data(self, csv_path: str, n_samples: int = 10) -> pd.DataFrame:
        """Load email dataset and sample n emails for testing."""
        df = pd.read_csv(csv_path)
        # Sample diverse emails
        sample_df = df.sample(n=min(n_samples, len(df)), random_state=42)
        return sample_df.reset_index(drop=True)
    
    def analyze_sentiment_v1(self, email_subject: str, email_body: str) -> Dict:
        """
        Initial sentiment analysis prompt (v1).
        """
        prompt = f"""Analyze the sentiment of this customer support email.

Email Subject: {email_subject}
Email Body: {email_body}

Classify the sentiment as:
- positive: Customer is happy, satisfied, or expressing gratitude
- negative: Customer is frustrated, angry, or reporting problems
- neutral: Factual, informational, or neither clearly positive nor negative

Respond with JSON:
{{
    "sentiment": "positive|negative|neutral",
    "confidence": 0.0-1.0,
    "reasoning": "explain your classification"
}}"""
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a sentiment analysis expert. Always respond with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            return {
                'sentiment': result.get('sentiment', 'neutral'),
                'confidence': float(result.get('confidence', 0.5)),
                'reasoning': result.get('reasoning', '')
            }
        except Exception as e:
            return {
                'sentiment': 'neutral',
                'confidence': 0.0,
                'reasoning': f'Error: {str(e)}'
            }
    
    def analyze_sentiment_v2(self, email_subject: str, email_body: str) -> Dict:
        """
        Improved sentiment analysis prompt (v2) with better guidelines.
        """
        prompt = f"""Analyze the sentiment of this customer support email with careful attention to context.

Email Subject: {email_subject}
Email Body: {email_body}

Guidelines for classification:
1. POSITIVE: 
   - Expressions of gratitude, satisfaction, or appreciation
   - Feature requests that are polite and constructive
   - Positive feedback or compliments
   - Examples: "Thank you", "Great feature", "Love this", "Helpful"

2. NEGATIVE:
   - Frustration, anger, or complaints
   - Reports of bugs, errors, or broken functionality
   - Urgent issues or problems affecting work
   - Examples: "Not working", "Error", "Broken", "Frustrated", "Issue"

3. NEUTRAL:
   - Factual questions or information requests
   - Setup or configuration help requests
   - Informational queries without emotional tone
   - Examples: "How do I...", "Can you help with...", "Need documentation"

Important considerations:
- Feature requests can be NEUTRAL if they're polite inquiries, not complaints
- Bug reports are typically NEGATIVE even if politely worded
- Questions about setup/configuration are usually NEUTRAL
- Consider the overall tone, not just individual words

Respond with JSON:
{{
    "sentiment": "positive|negative|neutral",
    "confidence": 0.0-1.0,
    "reasoning": "detailed explanation of classification, including key phrases that influenced the decision"
}}"""
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a sentiment analysis expert. Always respond with valid JSON. Be consistent in your classifications."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            return {
                'sentiment': result.get('sentiment', 'neutral'),
                'confidence': float(result.get('confidence', 0.5)),
                'reasoning': result.get('reasoning', '')
            }
        except Exception as e:
            return {
                'sentiment': 'neutral',
                'confidence': 0.0,
                'reasoning': f'Error: {str(e)}'
            }
    
    def evaluate_prompt(self, df: pd.DataFrame, version: str = 'v1') -> pd.DataFrame:
        """Evaluate a prompt version on the dataset."""
        results = []
        
        for idx, row in df.iterrows():
            if version == 'v1':
                result = self.analyze_sentiment_v1(row['subject'], row['body'])
            else:
                result = self.analyze_sentiment_v2(row['subject'], row['body'])
            
            results.append({
                'email_id': row['email_id'],
                'subject': row['subject'],
                'body': row['body'],
                'sentiment': result['sentiment'],
                'confidence': result['confidence'],
                'reasoning': result['reasoning']
            })
        
        return pd.DataFrame(results)
    
    def manual_evaluation(self, results_df: pd.DataFrame) -> Dict:
        """
        Manual evaluation framework.
        In production, this would be done by human evaluators.
        For this assignment, we'll use heuristics to simulate manual evaluation.
        """
        # Heuristic-based evaluation (in real scenario, this would be human-labeled)
        evaluation = {
            'consistent': 0,
            'inconsistent': 0,
            'low_confidence': 0,
            'unclear_reasoning': 0,
            'issues': []
        }
        
        for idx, row in results_df.iterrows():
            # Check confidence
            if row['confidence'] < 0.6:
                evaluation['low_confidence'] += 1
                evaluation['issues'].append({
                    'email_id': row['email_id'],
                    'issue': 'Low confidence score',
                    'confidence': row['confidence']
                })
            
            # Check reasoning quality
            if len(row['reasoning']) < 20:
                evaluation['unclear_reasoning'] += 1
                evaluation['issues'].append({
                    'email_id': row['email_id'],
                    'issue': 'Unclear or too brief reasoning',
                    'reasoning_length': len(row['reasoning'])
                })
            
            # Check for edge cases (feature requests, bug reports)
            text = (row['subject'] + ' ' + row['body']).lower()
            if 'feature request' in text or 'request' in text:
                if row['sentiment'] not in ['neutral', 'positive']:
                    evaluation['inconsistent'] += 1
                    evaluation['issues'].append({
                        'email_id': row['email_id'],
                        'issue': 'Feature request misclassified',
                        'predicted': row['sentiment']
                    })
            
            if any(word in text for word in ['error', 'bug', 'broken', 'not working', 'issue', 'problem']):
                if row['sentiment'] != 'negative':
                    evaluation['inconsistent'] += 1
                    evaluation['issues'].append({
                        'email_id': row['email_id'],
                        'issue': 'Problem report not classified as negative',
                        'predicted': row['sentiment']
                    })
        
        evaluation['consistent'] = len(results_df) - evaluation['inconsistent']
        
        return evaluation
    
    def compare_versions(self, v1_results: pd.DataFrame, v2_results: pd.DataFrame) -> Dict:
        """Compare results between v1 and v2 prompts."""
        comparison = {
            'sentiment_changes': 0,
            'confidence_improvements': 0,
            'changes': []
        }
        
        for idx in range(len(v1_results)):
            v1_row = v1_results.iloc[idx]
            v2_row = v2_results.iloc[idx]
            
            if v1_row['sentiment'] != v2_row['sentiment']:
                comparison['sentiment_changes'] += 1
                comparison['changes'].append({
                    'email_id': v1_row['email_id'],
                    'v1_sentiment': v1_row['sentiment'],
                    'v2_sentiment': v2_row['sentiment'],
                    'v1_confidence': v1_row['confidence'],
                    'v2_confidence': v2_row['confidence']
                })
            
            if v2_row['confidence'] > v1_row['confidence']:
                comparison['confidence_improvements'] += 1
        
        return comparison


def main():
    """Main execution function."""
    print("=" * 60)
    print("Part B: Sentiment Analysis Prompt Evaluation")
    print("=" * 60)
    
    # Initialize analyzer
    analyzer = SentimentAnalyzer()
    
    # Load test data (10 emails)
    print("\n1. Loading test dataset (10 emails)...")
    test_df = analyzer.load_data('dataset.csv', n_samples=10)
    print(f"   Loaded {len(test_df)} emails for testing")
    
    # Test Prompt v1
    print("\n2. Testing Prompt v1...")
    v1_results = analyzer.evaluate_prompt(test_df, version='v1')
    v1_eval = analyzer.manual_evaluation(v1_results)
    
    print(f"\n   Prompt v1 Results:")
    print(f"   - Consistent classifications: {v1_eval['consistent']}/{len(test_df)}")
    print(f"   - Inconsistent: {v1_eval['inconsistent']}")
    print(f"   - Low confidence: {v1_eval['low_confidence']}")
    print(f"   - Unclear reasoning: {v1_eval['unclear_reasoning']}")
    
    # Show sample results
    print("\n   Sample v1 Results (first 3):")
    for idx, row in v1_results.head(3).iterrows():
        print(f"\n   Email {row['email_id']}:")
        print(f"   Subject: {row['subject']}")
        print(f"   Sentiment: {row['sentiment']} (confidence: {row['confidence']:.2f})")
        print(f"   Reasoning: {row['reasoning'][:100]}...")
    
    # Test Prompt v2
    print("\n3. Testing Improved Prompt v2...")
    v2_results = analyzer.evaluate_prompt(test_df, version='v2')
    v2_eval = analyzer.manual_evaluation(v2_results)
    
    print(f"\n   Prompt v2 Results:")
    print(f"   - Consistent classifications: {v2_eval['consistent']}/{len(test_df)}")
    print(f"   - Inconsistent: {v2_eval['inconsistent']}")
    print(f"   - Low confidence: {v2_eval['low_confidence']}")
    print(f"   - Unclear reasoning: {v2_eval['unclear_reasoning']}")
    
    # Compare versions
    print("\n4. Comparing v1 vs v2:")
    comparison = analyzer.compare_versions(v1_results, v2_results)
    print(f"   - Sentiment changes: {comparison['sentiment_changes']}")
    print(f"   - Confidence improvements: {comparison['confidence_improvements']}")
    
    if comparison['changes']:
        print("\n   Changes detected:")
        for change in comparison['changes'][:3]:
            print(f"     Email {change['email_id']}: {change['v1_sentiment']} -> {change['v2_sentiment']}")
            print(f"       Confidence: {change['v1_confidence']:.2f} -> {change['v2_confidence']:.2f}")
    
    # Save results
    print("\n5. Saving results...")
    v1_results.to_csv('sentiment_results_v1.csv', index=False)
    v2_results.to_csv('sentiment_results_v2.csv', index=False)
    print("   Saved: sentiment_results_v1.csv, sentiment_results_v2.csv")
    
    print("\n" + "=" * 60)
    print("Part B Complete!")
    print("=" * 60)
    print("\nNext: Review the results and see part_b_report.md for detailed analysis.")


if __name__ == "__main__":
    main()

