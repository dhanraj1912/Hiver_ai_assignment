# Part B: Sentiment Analysis Prompt Evaluation Report

## Overview
This report documents the evaluation of two sentiment analysis prompts (v1 and v2) for customer support emails, focusing on consistency, accuracy, and debuggability.

## Prompt v1 - Initial Version

### Approach
The initial prompt was straightforward, asking the model to classify emails as positive, negative, or neutral with basic definitions.

### What Failed

1. **Inconsistent Classification of Feature Requests**
   - Feature requests were sometimes classified as negative (when they should be neutral or positive)
   - Example: "Feature request: bulk tagging" was occasionally marked negative due to the word "request" being interpreted as a complaint

2. **Ambiguity in Bug Reports**
   - Bug reports with polite language were sometimes classified as neutral instead of negative
   - Example: "Email threads not merging" - polite but reporting a problem should be negative

3. **Low Confidence on Edge Cases**
   - Emails with mixed tones received low confidence scores
   - Reasoning was often too brief to help with debugging

4. **Lack of Context Guidelines**
   - No specific guidance on how to handle common support email patterns
   - Model had to infer context without explicit examples

## Prompt v2 - Improved Version

### What Was Improved

1. **Explicit Classification Guidelines**
   - Added clear examples for each sentiment category
   - Provided specific word/phrase examples to guide classification

2. **Context-Aware Rules**
   - Explicitly stated that bug reports are typically negative even if politely worded
   - Clarified that feature requests can be neutral if polite
   - Added guidance on setup/configuration questions (usually neutral)

3. **Enhanced Reasoning Requirements**
   - Prompt now asks for "detailed explanation" including "key phrases that influenced the decision"
   - This improves debuggability by making the model's reasoning more transparent

4. **Better Edge Case Handling**
   - Added "Important considerations" section addressing common ambiguities
   - Helps the model make more consistent decisions on borderline cases

### Improvements Observed

- **Consistency**: Reduced misclassification of feature requests and bug reports
- **Confidence**: Higher confidence scores on clear-cut cases
- **Reasoning Quality**: More detailed explanations that help identify why classifications were made
- **Edge Cases**: Better handling of mixed-tone emails

## How to Evaluate Prompts Systematically

### 1. **Consistency Testing**
   - Run the same prompt multiple times on the same emails
   - Measure variance in outputs (should be minimal)
   - Track sentiment changes between runs

### 2. **Edge Case Evaluation**
   - Create a test set with known edge cases:
     - Polite bug reports
     - Feature requests
     - Mixed-tone emails
     - Sarcastic or ambiguous language
   - Measure accuracy on these cases

### 3. **Confidence Calibration**
   - Analyze confidence scores vs. actual accuracy
   - High confidence should correlate with correct classifications
   - Low confidence should flag cases needing human review

### 4. **Reasoning Quality Assessment**
   - Evaluate whether reasoning explains the classification
   - Check if reasoning mentions relevant key phrases
   - Ensure reasoning is actionable for debugging

### 5. **A/B Testing Framework**
   - Compare multiple prompt versions on the same dataset
   - Track metrics: accuracy, consistency, confidence distribution
   - Use statistical tests to determine if improvements are significant

### 6. **Human Evaluation**
   - Have human evaluators label a subset of emails
   - Compare model outputs to human labels
   - Identify systematic biases or patterns in errors

### 7. **Error Analysis**
   - Group errors by type (false positives, false negatives, misclassifications)
   - Analyze common failure patterns
   - Use insights to refine prompts iteratively

### 8. **Production Monitoring**
   - Track sentiment distribution over time
   - Monitor for sudden shifts that might indicate prompt drift
   - Collect user feedback on sentiment accuracy

## Recommendations for Production

1. **Use Prompt v2** with the enhanced guidelines
2. **Implement confidence thresholds**: Flag low-confidence predictions for human review
3. **Continuous evaluation**: Regularly test on new emails to catch edge cases
4. **Feedback loop**: Collect corrections and use them to refine the prompt
5. **Version control**: Track prompt versions and their performance metrics
6. **Monitoring**: Set up alerts for unusual sentiment distributions

## Conclusion

The improved prompt (v2) addresses the key failures of v1 by providing explicit guidelines, better context handling, and enhanced reasoning requirements. Systematic evaluation using the framework above will help maintain and improve prompt quality over time.

