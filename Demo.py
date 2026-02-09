"""
End-to-End Demo: Research Assistant RAG Pipeline

Demonstrates complete workflow:
1. Fetch papers from ArXiv/PubMed
2. Build FAISS vector index
3. Query with semantic search
4. Generate citation-grounded responses
"""

from research_assistant import ResearchAssistant
from api_connectors import ArXivConnector, PubMedConnector, MultiSourceConnector


def main():
    """
    Complete demo of research assistant capabilities
    """
    
    print("="*80)
    print("🔬 RESEARCH ASSISTANT - END-TO-END DEMO")
    print("="*80)
    print()
    
    # ========================================================================
    # STEP 1: Initialize API Connectors (Modular Architecture)
    # ========================================================================
    print("STEP 1: Initializing Modular API Connectors")
    print("-" * 80)
    
    arxiv = ArXivConnector()
    pubmed = PubMedConnector(email="your.email@example.com")
    multi_source = MultiSourceConnector([arxiv, pubmed])
    
    print("✅ ArXiv connector ready")
    print("✅ PubMed connector ready")
    print("✅ Multi-source aggregator ready\n")
    
    # ========================================================================
    # STEP 2: Fetch Papers from Multiple Sources
    # ========================================================================
    print("STEP 2: Fetching Papers from ArXiv and PubMed")
    print("-" * 80)
    
    # Example research topics
    topics = [
        "transformer neural networks",
        "CRISPR gene editing",
        "quantum computing algorithms"
    ]
    
    all_papers = []
    for topic in topics:
        print(f"\n📚 Searching for: '{topic}'")
        papers = multi_source.search_all(topic, max_results_per_source=5)
        all_papers.extend(papers)
    
    print(f"\n✅ Total papers collected: {len(all_papers)}")
    print(f"   - ArXiv papers: {sum(1 for p in all_papers if p.source == 'arxiv')}")
    print(f"   - PubMed papers: {sum(1 for p in all_papers if p.source == 'pubmed')}\n")
    
    # ========================================================================
    # STEP 3: Initialize Research Assistant with 4-bit Quantization
    # ========================================================================
    print("STEP 3: Initializing Research Assistant")
    print("-" * 80)
    
    assistant = ResearchAssistant(
        embedding_model_name="sentence-transformers/all-MiniLM-L6-v2",
        llm_model_name="meta-llama/Llama-2-7b-chat-hf",
        use_4bit_quantization=True  # Enable 4-bit quantization for GPU efficiency
    )
    
    # ========================================================================
    # STEP 4: Build FAISS Vector Index
    # ========================================================================
    print("\nSTEP 4: Building FAISS Vector Index")
    print("-" * 80)
    
    # Add papers to vector database
    # This creates embeddings and indexes them with FAISS
    assistant.add_papers(all_papers)
    
    # Save index for future use
    assistant.save_index()
    
    # ========================================================================
    # STEP 5: Query the System (RAG Pipeline)
    # ========================================================================
    print("\nSTEP 5: Running RAG Pipeline Queries")
    print("-" * 80)
    
    # Example research questions
    questions = [
        "How do transformers work in deep learning?",
        "What are the applications of CRISPR in gene therapy?",
        "What quantum algorithms show advantages over classical computing?"
    ]
    
    for i, question in enumerate(questions, 1):
        print(f"\n{'='*80}")
        print(f"QUERY {i}")
        print('='*80)
        
        # Run complete RAG pipeline
        result = assistant.query(question, top_k=5)
        
        # Display results
        print(f"\n📝 ANSWER:")
        print(result["answer"])
        
        print(f"\n📚 CITATIONS ({len(result['citations'])} papers):")
        for j, citation in enumerate(result["citations"], 1):
            print(f"\n{j}. {citation['title']}")
            print(f"   Authors: {', '.join(citation['authors'][:3])}")
            print(f"   Source: {citation['source'].upper()}")
            print(f"   Relevance: {citation['relevance_score']:.2%}")
            print(f"   URL: {citation['url']}")
        
        print(f"\n⚡ Processing time: {result['processing_time']:.2f}s")
    
    # ========================================================================
    # STEP 6: Demonstrate Sub-Second Performance
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 6: Performance Benchmark")
    print("-" * 80)
    
    import time
    
    # Test 10 rapid queries
    test_queries = [
        "What are attention mechanisms?",
        "How does CRISPR work?",
        "Explain quantum superposition",
        "What is deep learning?",
        "Applications of gene editing"
    ]
    
    times = []
    for query in test_queries:
        start = time.time()
        result = assistant.query(query, top_k=3)
        elapsed = time.time() - start
        times.append(elapsed)
    
    avg_time = sum(times) / len(times)
    print(f"\n📊 Performance Metrics:")
    print(f"   - Average query time: {avg_time:.3f}s")
    print(f"   - Fastest query: {min(times):.3f}s")
    print(f"   - Slowest query: {max(times):.3f}s")
    print(f"   - Sub-second queries: {sum(1 for t in times if t < 1.0)}/{len(times)}")
    
    if avg_time < 1.0:
        print(f"\n✅ ACHIEVED SUB-SECOND PROCESSING!")
    
    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "="*80)
    print("📊 SYSTEM SUMMARY")
    print("="*80)
    
    print(f"""
✅ Papers indexed: {len(all_papers)}
✅ Vector database: FAISS (L2 distance)
✅ Embedding model: SentenceTransformers
✅ LLM: Llama-2-7b (4-bit quantized)
✅ Data sources: ArXiv + PubMed
✅ Architecture: Modular connectors
✅ Response type: Citation-grounded
✅ Average query time: {avg_time:.3f}s
    """)
    
    print("="*80)
    print("🎉 DEMO COMPLETE!")
    print("="*80)


if __name__ == "__main__":
    main()
