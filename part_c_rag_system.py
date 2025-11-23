"""
Part C: Mini-RAG for Knowledge Base Answering
Builds a simple RAG system using embeddings for KB article retrieval.
"""

import os
import json
from typing import List, Dict, Tuple
from pathlib import Path
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

class MiniRAG:
    """
    Simple RAG system for knowledge base answering.
    Uses embeddings for retrieval and LLM for answer generation.
    """
    
    def __init__(self, api_key: str = None, use_openai: bool = False):
        """
        Initialize RAG system.
        
        Args:
            api_key: OpenAI API key (optional if using open-source embeddings)
            use_openai: If True, use OpenAI embeddings. If False, use sentence-transformers.
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.use_openai = use_openai and self.api_key is not None
        
        if self.use_openai:
            self.client = OpenAI(api_key=self.api_key)
            self.embedding_model = "text-embedding-3-small"
        else:
            # Use open-source embeddings
            print("Loading sentence-transformers model...")
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            if self.api_key:
                self.client = OpenAI(api_key=self.api_key)
            else:
                self.client = None
        
        # Initialize ChromaDB for vector storage
        self.client_db = chromadb.Client(Settings(anonymized_telemetry=False))
        self.collection = None
    
    def load_kb_articles(self, kb_folder: str) -> List[Dict]:
        """Load knowledge base articles from folder."""
        articles = []
        kb_path = Path(kb_folder)
        
        if not kb_path.exists():
            raise ValueError(f"KB folder not found: {kb_folder}")
        
        # Load all .txt files in the folder
        for file_path in kb_path.glob("*.txt"):
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                articles.append({
                    'id': file_path.stem,
                    'title': file_path.stem.replace('_', ' ').title(),
                    'content': content,
                    'file_path': str(file_path)
                })
        
        return articles
    
    def get_embedding(self, text: str) -> List[float]:
        """Get embedding for text."""
        if self.use_openai:
            response = self.client.embeddings.create(
                model=self.embedding_model,
                input=text
            )
            return response.data[0].embedding
        else:
            return self.embedding_model.encode(text).tolist()
    
    def build_index(self, articles: List[Dict]):
        """Build vector index from KB articles."""
        print(f"Building index for {len(articles)} articles...")
        
        # Create or get collection
        try:
            self.collection = self.client_db.get_collection("kb_articles")
            self.client_db.delete_collection("kb_articles")
        except:
            pass
        
        self.collection = self.client_db.create_collection("kb_articles")
        
        # Add articles to collection
        ids = []
        documents = []
        metadatas = []
        embeddings = []
        
        for article in articles:
            ids.append(article['id'])
            documents.append(article['content'])
            metadatas.append({'title': article['title']})
            embeddings.append(self.get_embedding(article['content']))
        
        self.collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings
        )
        
        print(f"Index built successfully!")
    
    def retrieve(self, query: str, top_k: int = 3) -> List[Dict]:
        """Retrieve relevant articles for a query."""
        query_embedding = self.get_embedding(query)
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        
        retrieved = []
        if results['ids'] and len(results['ids'][0]) > 0:
            for i in range(len(results['ids'][0])):
                retrieved.append({
                    'id': results['ids'][0][i],
                    'title': results['metadatas'][0][i]['title'],
                    'content': results['documents'][0][i],
                    'distance': results['distances'][0][i] if 'distances' in results else None
                })
        
        return retrieved
    
    def generate_answer(
        self, query: str, retrieved_articles: List[Dict]
    ) -> Tuple[str, float]:
        """
        Generate answer from retrieved articles.
        Returns: (answer, confidence_score)
        """
        if not self.client:
            # Fallback if no OpenAI API key
            return self._generate_answer_fallback(query, retrieved_articles)
        
        # Build context from retrieved articles
        context = "\n\n".join([
            f"Article: {article['title']}\n{article['content']}"
            for article in retrieved_articles
        ])
        
        prompt = f"""You are a helpful assistant answering questions about Hiver based on knowledge base articles.

Question: {query}

Relevant Knowledge Base Articles:
{context}

Instructions:
- Answer the question based ONLY on the provided articles
- If the answer is not in the articles, say so clearly
- Be concise and accurate
- Cite which article(s) you used

Respond with JSON:
{{
    "answer": "your answer here",
    "confidence": 0.0-1.0,
    "sources": ["article_id1", "article_id2"]
}}"""
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant. Always respond with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            
            # Calculate confidence based on retrieval quality and model confidence
            base_confidence = float(result.get('confidence', 0.5))
            retrieval_quality = self._calculate_retrieval_quality(retrieved_articles)
            final_confidence = (base_confidence * 0.7 + retrieval_quality * 0.3)
            
            return result.get('answer', ''), final_confidence
        except Exception as e:
            return f"Error generating answer: {str(e)}", 0.0
    
    def _generate_answer_fallback(self, query: str, retrieved_articles: List[Dict]) -> Tuple[str, float]:
        """Fallback answer generation without OpenAI."""
        if not retrieved_articles:
            return "No relevant articles found.", 0.0
        
        # Simple extraction-based answer
        best_article = retrieved_articles[0]
        confidence = 1.0 - (best_article.get('distance', 0.5) if best_article.get('distance') else 0.5)
        confidence = max(0.0, min(1.0, confidence))
        
        answer = f"Based on the article '{best_article['title']}':\n{best_article['content'][:500]}..."
        return answer, confidence
    
    def _calculate_retrieval_quality(self, retrieved_articles: List[Dict]) -> float:
        """Calculate quality score based on retrieval distances."""
        if not retrieved_articles:
            return 0.0
        
        # Convert distances to similarity scores (assuming cosine distance)
        # Lower distance = higher similarity
        distances = [a.get('distance', 1.0) for a in retrieved_articles if a.get('distance') is not None]
        if not distances:
            return 0.5  # Default if no distance info
        
        # Average similarity (1 - distance)
        avg_similarity = 1.0 - (sum(distances) / len(distances))
        return max(0.0, min(1.0, avg_similarity))
    
    def query(self, question: str, top_k: int = 3) -> Dict:
        """
        Complete RAG query: retrieve + generate.
        Returns: {
            'query': question,
            'retrieved_articles': [...],
            'answer': ...,
            'confidence': ...
        }
        """
        print(f"\nQuery: {question}")
        print("-" * 60)
        
        # Retrieve
        retrieved = self.retrieve(question, top_k=top_k)
        print(f"\nRetrieved {len(retrieved)} articles:")
        for i, article in enumerate(retrieved, 1):
            print(f"\n{i}. {article['title']} (ID: {article['id']})")
            if article.get('distance'):
                print(f"   Similarity: {1 - article['distance']:.3f}")
            print(f"   Preview: {article['content'][:150]}...")
        
        # Generate answer
        answer, confidence = self.generate_answer(question, retrieved)
        
        print(f"\nGenerated Answer:")
        print(f"{answer}")
        print(f"\nConfidence Score: {confidence:.2%}")
        
        return {
            'query': question,
            'retrieved_articles': retrieved,
            'answer': answer,
            'confidence': confidence
        }


def create_sample_kb_articles(kb_folder: str = "kb_articles"):
    """Create sample knowledge base articles."""
    Path(kb_folder).mkdir(exist_ok=True)
    
    articles = {
        "automation_configuration": """How to Configure Automations in Hiver

Automations in Hiver help you streamline your email workflow by automatically performing actions based on triggers.

Setting Up an Automation:
1. Go to Settings > Automations
2. Click "Create New Automation"
3. Choose a trigger:
   - When email arrives
   - When email is tagged
   - When email status changes
   - When email is assigned
4. Set conditions (optional):
   - From specific email addresses
   - With specific tags
   - In specific shared mailboxes
5. Choose actions:
   - Assign to user/team
   - Add tags
   - Change status
   - Send notification
   - Create task
6. Click "Save"

Best Practices:
- Test automations before activating
- Use clear naming conventions
- Review automation logs regularly
- Start with simple automations and add complexity gradually

Troubleshooting:
- If automation doesn't trigger, check conditions
- Verify user permissions
- Check automation logs in Settings > Automations > Logs""",
        
        "csat_visibility": """CSAT (Customer Satisfaction) Visibility Issues

If CSAT scores are not appearing in your dashboard, check the following:

1. CSAT Survey Configuration:
   - Go to Settings > CSAT
   - Ensure "Enable CSAT Surveys" is turned ON
   - Verify survey is set to send automatically after email resolution

2. Email Resolution Status:
   - CSAT surveys only send for emails marked as "Resolved"
   - Check that emails are properly resolved before expecting surveys

3. Customer Response:
   - CSAT scores only appear after customers respond to surveys
   - Allow 24-48 hours for customer responses

4. Dashboard Settings:
   - Go to Analytics > CSAT Dashboard
   - Check date range filters
   - Verify you have permission to view CSAT data

5. Common Issues:
   - Surveys not sending: Check email deliverability
   - Scores not updating: Refresh dashboard or check sync status
   - Missing historical data: CSAT only tracks from when it was enabled

If issues persist:
- Check CSAT logs in Settings > CSAT > Logs
- Verify email addresses are valid
- Contact support if surveys are not being delivered""",
        
        "tagging_system": """Email Tagging System in Hiver

Tags help organize and categorize emails for better tracking and analytics.

Creating Tags:
1. Go to Settings > Tags
2. Click "Create New Tag"
3. Enter tag name and choose color
4. Set tag visibility (team-wide or personal)

Applying Tags:
- Manual: Click tag icon on email and select tags
- Automatic: Use automations to auto-tag based on conditions
- Bulk: Select multiple emails and apply tags

Tag Management:
- Edit tags: Settings > Tags > Edit
- Delete tags: Settings > Tags > Delete (removes from all emails)
- Tag permissions: Control who can create/edit tags

Best Practices:
- Use consistent naming conventions
- Limit number of tags per email (recommended: 3-5)
- Use hierarchical tags for better organization""",
        
        "shared_mailbox_access": """Shared Mailbox Access and Permissions

Shared mailboxes allow teams to collaborate on customer emails.

Accessing Shared Mailboxes:
1. Go to Mailboxes in left sidebar
2. Click on shared mailbox name
3. If you don't see it, request access from admin

Permission Levels:
- Viewer: Can read emails only
- Agent: Can read and reply
- Admin: Full access including settings

Troubleshooting Access Issues:
- "Permission denied" error: Contact admin to grant access
- Mailbox not visible: Check if you're added to the mailbox
- Can't reply: Verify you have Agent or Admin permissions

Requesting Access:
- Contact your team admin
- Or go to Settings > Mailboxes > Request Access""",
        
        "workflow_rules": """Workflow Rules Configuration

Workflow rules automate email handling based on conditions.

Creating Rules:
1. Settings > Rules
2. Click "New Rule"
3. Set conditions (from, subject contains, has tag, etc.)
4. Set actions (assign, tag, forward, etc.)
5. Save and activate

Rule Priority:
- Rules execute in order of creation
- First matching rule applies
- Reorder rules in Settings > Rules

Common Rule Types:
- Auto-assignment: Assign emails to team members
- Auto-tagging: Tag emails based on content
- Auto-forwarding: Forward to external addresses
- Status changes: Change email status automatically

Troubleshooting:
- Rule not firing: Check conditions match
- Multiple rules conflicting: Review rule order
- Test rules using "Test Rule" feature""",
        
        "sla_configuration": """SLA (Service Level Agreement) Configuration

SLAs help track response and resolution times for customer emails.

Setting Up SLAs:
1. Go to Settings > SLA
2. Click "Create SLA Rule"
3. Configure:
   - Response time target (e.g., 2 hours)
   - Resolution time target (e.g., 24 hours)
   - Business hours (when SLA clock runs)
   - Customer tiers (VIP, Standard, etc.)
4. Apply to specific mailboxes or tags
5. Save and activate

SLA Tracking:
- View SLA status in email list (color indicators)
- Check SLA dashboard in Analytics
- Set up alerts for approaching deadlines

Best Practices:
- Set realistic targets based on team capacity
- Different SLAs for different customer tiers
- Regular review of SLA performance
- Adjust based on actual performance data""",
        
        "email_threading": """Email Threading and Conversation Management

Hiver groups related emails into threads for better organization.

Threading Behavior:
- Emails with same subject line are grouped
- Replies automatically thread to original email
- Threads show chronological conversation

Thread Management:
- Split thread: Separate emails into different threads
- Merge threads: Combine related conversations
- View thread history: Click to expand full conversation

Troubleshooting Threading Issues:
- Emails not threading: Check subject line matches
- Wrong threading: Use "Split Thread" to separate
- Missing emails in thread: Check filters and date range""",
        
        "analytics_dashboard": """Analytics and Reporting in Hiver

Analytics help track team performance and email metrics.

Key Metrics:
- Response time: Average time to first response
- Resolution time: Average time to resolve
- Email volume: Emails received/responded
- CSAT scores: Customer satisfaction ratings
- Agent performance: Individual team member stats

Accessing Analytics:
1. Go to Analytics in main menu
2. Choose dashboard (Team, Individual, CSAT, etc.)
3. Set date range and filters
4. Export reports as needed

Custom Reports:
- Create custom date ranges
- Filter by mailbox, tag, agent
- Export to CSV for external analysis

Dashboard Settings:
- Customize visible metrics
- Set default date ranges
- Configure refresh intervals"""
    }
    
    for filename, content in articles.items():
        file_path = Path(kb_folder) / f"{filename}.txt"
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
    
    print(f"Created {len(articles)} KB articles in {kb_folder}/")
    return articles


def main():
    """Main execution function."""
    print("=" * 60)
    print("Part C: Mini-RAG for Knowledge Base Answering")
    print("=" * 60)
    
    # Create sample KB articles if they don't exist
    kb_folder = "kb_articles"
    if not Path(kb_folder).exists() or len(list(Path(kb_folder).glob("*.txt"))) == 0:
        print("\n1. Creating sample KB articles...")
        create_sample_kb_articles(kb_folder)
    else:
        print(f"\n1. Using existing KB articles in {kb_folder}/")
    
    # Initialize RAG system
    print("\n2. Initializing RAG system...")
    use_openai = os.getenv("OPENAI_API_KEY") is not None
    rag = MiniRAG(use_openai=use_openai)
    
    # Load and index articles
    print("\n3. Loading and indexing KB articles...")
    articles = rag.load_kb_articles(kb_folder)
    print(f"   Loaded {len(articles)} articles")
    rag.build_index(articles)
    
    # Query 1
    print("\n" + "=" * 60)
    print("QUERY 1: How do I configure automations in Hiver?")
    print("=" * 60)
    result1 = rag.query("How do I configure automations in Hiver?", top_k=3)
    
    # Query 2
    print("\n" + "=" * 60)
    print("QUERY 2: Why is CSAT not appearing?")
    print("=" * 60)
    result2 = rag.query("Why is CSAT not appearing?", top_k=3)
    
    # Save results
    print("\n" + "=" * 60)
    print("Saving results...")
    with open('rag_results.json', 'w') as f:
        json.dump({
            'query1': {
                'query': result1['query'],
                'retrieved_article_ids': [a['id'] for a in result1['retrieved_articles']],
                'answer': result1['answer'],
                'confidence': result1['confidence']
            },
            'query2': {
                'query': result2['query'],
                'retrieved_article_ids': [a['id'] for a in result2['retrieved_articles']],
                'answer': result2['answer'],
                'confidence': result2['confidence']
            }
        }, f, indent=2)
    print("   Saved: rag_results.json")
    
    print("\n" + "=" * 60)
    print("Part C Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

