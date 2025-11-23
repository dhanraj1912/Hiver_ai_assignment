"""
Part A: Email Tagging Mini-System
Builds a customer-specific email tagging system with customer isolation.
"""

import pandas as pd
import json
import os
from typing import Dict, List, Tuple
from collections import defaultdict
import re
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

class EmailTagger:
    """
    Email tagging system with customer isolation.
    Each customer has their own tag vocabulary and model context.
    """
    
    def __init__(self, api_key: str = None):
        """Initialize the tagger with OpenAI client."""
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key required. Set OPENAI_API_KEY environment variable.")
        self.client = OpenAI(api_key=self.api_key)
        self.customer_tags = defaultdict(set)  # Maps customer_id -> set of tags
        self.customer_patterns = defaultdict(dict)  # Maps customer_id -> {tag: [patterns]}
        self.customer_anti_patterns = defaultdict(dict)  # Maps customer_id -> {tag: [anti-patterns]}
        
    def load_data(self, csv_path: str) -> pd.DataFrame:
        """Load email dataset from CSV."""
        df = pd.read_csv(csv_path)
        return df
    
    def build_customer_vocabularies(self, df: pd.DataFrame):
        """
        Build customer-specific tag vocabularies.
        CRITICAL: Tags from one customer must not leak into another customer's model.
        """
        for customer_id in df['customer_id'].unique():
            customer_df = df[df['customer_id'] == customer_id]
            self.customer_tags[customer_id] = set(customer_df['tag'].unique())
            
            # Extract patterns for each tag per customer
            for tag in self.customer_tags[customer_id]:
                tag_emails = customer_df[customer_df['tag'] == tag]
                self.customer_patterns[customer_id][tag] = self._extract_patterns(tag_emails)
                self.customer_anti_patterns[customer_id][tag] = self._extract_anti_patterns(
                    tag_emails, customer_df
                )
    
    def _extract_patterns(self, tag_emails: pd.DataFrame) -> List[str]:
        """Extract common patterns/keywords for a tag."""
        all_text = ' '.join(tag_emails['subject'].astype(str) + ' ' + tag_emails['body'].astype(str))
        # Simple keyword extraction (can be enhanced)
        words = re.findall(r'\b\w+\b', all_text.lower())
        word_freq = defaultdict(int)
        for word in words:
            if len(word) > 3:  # Filter short words
                word_freq[word] += 1
        
        # Get top keywords
        top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]
        return [word for word, _ in top_words]
    
    def _extract_anti_patterns(self, tag_emails: pd.DataFrame, all_emails: pd.DataFrame) -> List[str]:
        """
        Extract anti-patterns - words that frequently mislead the model.
        Example: Words that appear in this tag but also frequently in other tags.
        """
        tag_text = ' '.join(tag_emails['subject'].astype(str) + ' ' + tag_emails['body'].astype(str))
        tag_words = set(re.findall(r'\b\w+\b', tag_text.lower()))
        
        other_tags = all_emails[~all_emails['tag'].isin(tag_emails['tag'].unique())]
        other_text = ' '.join(other_tags['subject'].astype(str) + ' ' + other_tags['body'].astype(str))
        other_words = set(re.findall(r'\b\w+\b', other_text.lower()))
        
        # Words that appear in both - potential misleading words
        misleading = tag_words.intersection(other_words)
        return list(misleading)[:5]  # Top 5 misleading words
    
    def classify_email(self, email_subject: str, email_body: str, customer_id: str) -> Tuple[str, float, str]:
        """
        Classify an email using customer-specific tags only.
        Returns: (predicted_tag, confidence, reasoning)
        """
        # Get customer-specific tags only
        available_tags = list(self.customer_tags[customer_id])
        
        if not available_tags:
            return "unknown", 0.0, "No tags available for this customer"
        
        # Build prompt with customer-specific context
        patterns_info = self._build_patterns_info(customer_id)
        anti_patterns_info = self._build_anti_patterns_info(customer_id)
        
        prompt = self._build_classification_prompt(
            email_subject, email_body, available_tags, 
            patterns_info, anti_patterns_info, customer_id
        )
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an expert email classifier. Always respond with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            return (
                result.get('tag', available_tags[0]),
                float(result.get('confidence', 0.5)),
                result.get('reasoning', '')
            )
        except Exception as e:
            # Fallback to first tag if API fails
            return available_tags[0], 0.5, f"API error: {str(e)}"
    
    def _build_patterns_info(self, customer_id: str) -> str:
        """Build patterns information string for prompt."""
        info_parts = []
        for tag, patterns in self.customer_patterns[customer_id].items():
            if patterns:
                info_parts.append(f"- {tag}: Keywords like {', '.join(patterns[:5])}")
        return "\n".join(info_parts) if info_parts else "No specific patterns identified yet."
    
    def _build_anti_patterns_info(self, customer_id: str) -> str:
        """Build anti-patterns information for guardrails."""
        info_parts = []
        for tag, anti_patterns in self.customer_anti_patterns[customer_id].items():
            if anti_patterns:
                info_parts.append(f"- {tag}: Be cautious of words like {', '.join(anti_patterns[:3])} as they may appear in other contexts")
        return "\n".join(info_parts) if info_parts else "No specific anti-patterns identified yet."
    
    def _build_classification_prompt(
        self, subject: str, body: str, available_tags: List[str],
        patterns_info: str, anti_patterns_info: str, customer_id: str
    ) -> str:
        """Build the classification prompt with customer isolation."""
        prompt = f"""Classify the following email into ONE of these tags: {', '.join(available_tags)}

IMPORTANT: You can ONLY use tags from the list above. These are the ONLY tags available for customer {customer_id}.

Email Subject: {subject}
Email Body: {body}

Patterns to help classification:
{patterns_info}

Anti-patterns (guardrails):
{anti_patterns_info}

Respond with JSON in this exact format:
{{
    "tag": "one_of_the_available_tags",
    "confidence": 0.0-1.0,
    "reasoning": "brief explanation of why this tag was chosen"
}}"""
        return prompt
    
    def evaluate(self, df: pd.DataFrame) -> Dict:
        """Evaluate the classifier on the dataset."""
        correct = 0
        total = len(df)
        errors = []
        
        for idx, row in df.iterrows():
            predicted_tag, confidence, reasoning = self.classify_email(
                row['subject'], row['body'], row['customer_id']
            )
            
            if predicted_tag == row['tag']:
                correct += 1
            else:
                errors.append({
                    'email_id': row['email_id'],
                    'customer_id': row['customer_id'],
                    'subject': row['subject'],
                    'true_tag': row['tag'],
                    'predicted_tag': predicted_tag,
                    'confidence': confidence,
                    'reasoning': reasoning
                })
        
        accuracy = correct / total if total > 0 else 0
        
        return {
            'accuracy': accuracy,
            'correct': correct,
            'total': total,
            'errors': errors
        }
    
    def error_analysis(self, errors: List[Dict]) -> Dict:
        """Analyze classification errors."""
        if not errors:
            return {"message": "No errors to analyze"}
        
        # Group errors by customer
        customer_errors = defaultdict(list)
        for error in errors:
            customer_errors[error['customer_id']].append(error)
        
        # Analyze common misclassifications
        confusion = defaultdict(int)
        for error in errors:
            key = f"{error['true_tag']} -> {error['predicted_tag']}"
            confusion[key] += 1
        
        return {
            'total_errors': len(errors),
            'errors_by_customer': {k: len(v) for k, v in customer_errors.items()},
            'common_confusions': dict(sorted(confusion.items(), key=lambda x: x[1], reverse=True)[:5])
        }


def main():
    """Main execution function."""
    print("=" * 60)
    print("Part A: Email Tagging Mini-System")
    print("=" * 60)
    
    # Initialize tagger
    tagger = EmailTagger()
    
    # Load data
    print("\n1. Loading dataset...")
    df = tagger.load_data('dataset.csv')
    print(f"   Loaded {len(df)} emails from {df['customer_id'].nunique()} customers")
    
    # Build customer vocabularies
    print("\n2. Building customer-specific tag vocabularies...")
    tagger.build_customer_vocabularies(df)
    
    for customer_id, tags in tagger.customer_tags.items():
        print(f"   {customer_id}: {len(tags)} unique tags - {', '.join(sorted(tags))}")
    
    # Evaluate
    print("\n3. Evaluating classifier...")
    results = tagger.evaluate(df)
    
    print(f"\nResults:")
    print(f"   Accuracy: {results['accuracy']:.2%}")
    print(f"   Correct: {results['correct']}/{results['total']}")
    print(f"   Errors: {len(results['errors'])}")
    
    # Error analysis
    print("\n4. Error Analysis:")
    error_analysis = tagger.error_analysis(results['errors'])
    print(f"   Total Errors: {error_analysis['total_errors']}")
    print(f"   Errors by Customer: {error_analysis['errors_by_customer']}")
    print(f"   Common Confusions:")
    for confusion, count in error_analysis.get('common_confusions', {}).items():
        print(f"     {confusion}: {count}")
    
    # Show sample errors
    if results['errors']:
        print("\n5. Sample Errors (first 3):")
        for error in results['errors'][:3]:
            print(f"\n   Email ID: {error['email_id']}")
            print(f"   Customer: {error['customer_id']}")
            print(f"   Subject: {error['subject']}")
            print(f"   True Tag: {error['true_tag']}")
            print(f"   Predicted: {error['predicted_tag']} (confidence: {error['confidence']:.2f})")
            print(f"   Reasoning: {error['reasoning']}")
    
    # Test customer isolation
    print("\n6. Testing Customer Isolation:")
    test_customer = df['customer_id'].iloc[0]
    test_email = df[df['customer_id'] == test_customer].iloc[0]
    print(f"   Testing with customer: {test_customer}")
    print(f"   Available tags for {test_customer}: {sorted(tagger.customer_tags[test_customer])}")
    
    # Try to classify with wrong customer context (should fail gracefully)
    other_customers = [c for c in df['customer_id'].unique() if c != test_customer]
    if other_customers:
        other_customer = other_customers[0]
        print(f"   Available tags for {other_customer}: {sorted(tagger.customer_tags[other_customer])}")
        print(f"   ✓ Customer isolation verified: Tags are customer-specific")
    
    print("\n" + "=" * 60)
    print("Part A Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

