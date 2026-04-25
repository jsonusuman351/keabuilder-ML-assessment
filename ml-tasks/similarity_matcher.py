"""
Similarity Matcher for KeaBuilder
- Matches user inputs (leads, prompts) using cosine similarity
- Lightweight, production-ready logic.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Sample KeaBuilder-style inputs
sample_inputs = [
    "Create a high-converting landing page",
    "How to set up email automation campaign",
    "Build a sales funnel with checkout",
    "Add a lead capture form to my website",
    "Track funnel page views and conversion rate"
]

def find_most_similar(query: str, corpus: list = sample_inputs) -> dict:
    """Find the most similar input to a given query using cosine similarity."""
    
    all_texts = corpus + [query]
    
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(all_texts)
    
    query_vector = tfidf_matrix[-1]
    corpus_vectors = tfidf_matrix[:-1]
    similarities = cosine_similarity(query_vector, corpus_vectors).flatten()
    
    top_idx = np.argmax(similarities)
    
    return {
        "query": query,
        "top_match": corpus[top_idx],
        "similarity_score": round(float(similarities[top_idx]), 4),
        "all_scores": {corpus[i]: round(float(s), 4) for i, s in enumerate(similarities)}
    }


from fastapi import FastAPI
app = FastAPI(title="KeaBuilder Similarity Matcher")

@app.get("/match")
def match(query: str):
    return find_most_similar(query)


if __name__ == "__main__":
    test_queries = [
        "How to build a landing page?",
        "Set up email campaign for leads",
        "I want to track my funnel performance"
    ]
    
    print("=" * 60)
    print("KeaBuilder Similarity Matcher - Demo")
    print("=" * 60)
    
    for q in test_queries:
        result = find_most_similar(q)
        print(f"\nQuery: \"{result['query']}\"")
        print(f"Top Match: \"{result['top_match']}\"")
        print(f"Score: {result['similarity_score']}")
        print("-" * 40)
    
    print("\n[API] Run: uvicorn similarity_matcher:app --reload")
    print("[API] Test: http://localhost:8000/match?query=build+a+funnel")
