"""
Agentic AI Research Assistant
Built with RAG pipeline, FAISS vector search, and 4-bit quantized Llama
"""

import os
import numpy as np
from typing import List, Dict, Tuple
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer
import faiss
from dataclasses import dataclass
import json
import time


@dataclass
class Paper:
    """Data class for research papers"""
    title: str
    authors: List[str]
    abstract: str
    url: str
    source: str  # 'arxiv' or 'pubmed'
    paper_id: str
    

class ResearchAssistant:
    """
    Main research assistant class implementing RAG pipeline
    
    Architecture:
    1. Paper Retrieval (ArXiv/PubMed APIs) -> API Connectors
    2. Embedding Generation -> SentenceTransformers
    3. Vector Storage & Search -> FAISS
    4. Response Generation -> Quantized Llama
    5. Citation Grounding -> Custom formatting
    """
    
    def __init__(
        self,
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        llm_model_name: str = "meta-llama/Llama-2-7b-chat-hf",
        use_4bit_quantization: bool = True,
        faiss_index_path: str = "faiss_index.bin",
        papers_metadata_path: str = "papers_metadata.json"
    ):
        """
        Initialize the research assistant
        
        Args:
            embedding_model_name: SentenceTransformer model for embeddings
            llm_model_name: Llama model for text generation
            use_4bit_quantization: Enable 4-bit quantization for GPU efficiency
            faiss_index_path: Path to save/load FAISS index
            papers_metadata_path: Path to save/load paper metadata
        """
        print("🚀 Initializing Research Assistant...")
        
        # 1. Initialize SentenceTransformer for embeddings
        print("📊 Loading embedding model...")
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        
        # 2. Initialize FAISS index for vector search
        print("🔍 Initializing FAISS index...")
        self.faiss_index = faiss.IndexFlatL2(self.embedding_dim)  # L2 distance
        # For better performance with large datasets, use IndexIVFFlat:
        # quantizer = faiss.IndexFlatL2(self.embedding_dim)
        # self.faiss_index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, 100)
        
        # 3. Storage for paper metadata (titles, authors, URLs, etc.)
        self.papers: List[Paper] = []
        self.faiss_index_path = faiss_index_path
        self.papers_metadata_path = papers_metadata_path
        
        # 4. Initialize 4-bit quantized Llama for response generation
        print("🤖 Loading 4-bit quantized Llama model...")
        self.tokenizer, self.llm = self._load_quantized_llm(
            llm_model_name, use_4bit_quantization
        )
        
        print("✅ Research Assistant initialized!\n")
    
    def _load_quantized_llm(
        self, 
        model_name: str, 
        use_4bit: bool
    ) -> Tuple[AutoTokenizer, AutoModelForCausalLM]:
        """
        Load Llama model with 4-bit quantization for memory efficiency
        
        4-bit quantization reduces model size by ~75% with minimal quality loss
        Uses bitsandbytes library for efficient GPU inference
        """
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        if use_4bit:
            # Configure 4-bit quantization
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,  # Nested quantization for extra compression
                bnb_4bit_quant_type="nf4"  # Normal Float 4-bit quantization
            )
            
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                device_map="auto",  # Automatically distribute across GPUs
                torch_dtype=torch.float16
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                torch_dtype=torch.float16
            )
        
        return tokenizer, model
    
    def add_papers(self, papers: List[Paper]):
        """
        Add papers to the vector database
        
        Process:
        1. Generate embeddings for each paper's abstract
        2. Add embeddings to FAISS index
        3. Store paper metadata for retrieval
        """
        print(f"📚 Adding {len(papers)} papers to vector database...")
        
        # Generate embeddings for all abstracts
        abstracts = [paper.abstract for paper in papers]
        embeddings = self.embedding_model.encode(
            abstracts,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        # Add to FAISS index
        self.faiss_index.add(embeddings.astype('float32'))
        
        # Store paper metadata
        self.papers.extend(papers)
        
        print(f"✅ Total papers in database: {len(self.papers)}\n")
    
    def retrieve_papers(
        self, 
        query: str, 
        top_k: int = 5
    ) -> List[Tuple[Paper, float]]:
        """
        Retrieve most relevant papers for a query
        
        Process:
        1. Convert query to embedding
        2. FAISS searches for nearest neighbors (semantic similarity)
        3. Return top-k papers with relevance scores
        
        Args:
            query: User's research question
            top_k: Number of papers to retrieve
            
        Returns:
            List of (Paper, relevance_score) tuples
        """
        start_time = time.time()
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode(
            [query], 
            convert_to_numpy=True
        )
        
        # FAISS search - finds k nearest neighbors
        distances, indices = self.faiss_index.search(
            query_embedding.astype('float32'), 
            top_k
        )
        
        # Convert L2 distances to similarity scores (lower distance = higher similarity)
        # Normalize to 0-1 range
        max_distance = distances[0].max()
        relevance_scores = 1 - (distances[0] / max_distance) if max_distance > 0 else np.ones_like(distances[0])
        
        # Retrieve papers with scores
        results = [
            (self.papers[idx], float(score))
            for idx, score in zip(indices[0], relevance_scores)
        ]
        
        retrieval_time = time.time() - start_time
        print(f"⚡ Retrieved {top_k} papers in {retrieval_time:.3f}s")
        
        return results
    
    def generate_response(
        self,
        query: str,
        retrieved_papers: List[Tuple[Paper, float]],
        max_length: int = 512
    ) -> str:
        """
        Generate citation-grounded response using Llama
        
        Process:
        1. Construct prompt with retrieved papers as context
        2. Generate response using quantized Llama
        3. Ensure citations are included
        
        Args:
            query: User's question
            retrieved_papers: Papers retrieved from FAISS
            max_length: Maximum response length
            
        Returns:
            Generated response with citations
        """
        # Build context from retrieved papers
        context = self._build_context(retrieved_papers)
        
        # Construct prompt for Llama
        prompt = f"""You are a research assistant. Answer the question based ONLY on the provided research papers. Always cite your sources using [Paper X] format.

Context from research papers:
{context}

Question: {query}

Answer (with citations):"""
        
        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.llm.device)
        
        # Generate response
        with torch.no_grad():
            outputs = self.llm.generate(
                **inputs,
                max_new_tokens=max_length,
                temperature=0.7,
                top_p=0.9,
                do_sample=True
            )
        
        # Decode response
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract just the answer part (after the prompt)
        answer = response.split("Answer (with citations):")[1].strip()
        
        return answer
    
    def _build_context(self, retrieved_papers: List[Tuple[Paper, float]]) -> str:
        """Build formatted context from retrieved papers"""
        context_parts = []
        
        for i, (paper, score) in enumerate(retrieved_papers, 1):
            context_parts.append(
                f"[Paper {i}] {paper.title}\n"
                f"Authors: {', '.join(paper.authors)}\n"
                f"Abstract: {paper.abstract}\n"
                f"Relevance Score: {score:.2f}\n"
            )
        
        return "\n".join(context_parts)
    
    def query(self, question: str, top_k: int = 5) -> Dict:
        """
        End-to-end query processing (main entry point)
        
        Complete RAG pipeline:
        1. Retrieve relevant papers (FAISS)
        2. Generate citation-grounded response (Llama)
        3. Return formatted result
        
        Args:
            question: Research question
            top_k: Number of papers to retrieve
            
        Returns:
            Dictionary with answer, citations, and metadata
        """
        print(f"\n❓ Question: {question}")
        start_time = time.time()
        
        # Step 1: Retrieve papers
        retrieved_papers = self.retrieve_papers(question, top_k)
        
        # Step 2: Generate response
        print("🤖 Generating response...")
        answer = self.generate_response(question, retrieved_papers)
        
        # Step 3: Format citations
        citations = [
            {
                "title": paper.title,
                "authors": paper.authors,
                "url": paper.url,
                "source": paper.source,
                "relevance_score": score
            }
            for paper, score in retrieved_papers
        ]
        
        total_time = time.time() - start_time
        print(f"✅ Query completed in {total_time:.2f}s\n")
        
        return {
            "question": question,
            "answer": answer,
            "citations": citations,
            "processing_time": total_time,
            "num_papers_retrieved": len(retrieved_papers)
        }
    
    def save_index(self):
        """Save FAISS index and paper metadata to disk"""
        print("💾 Saving FAISS index and metadata...")
        
        # Save FAISS index
        faiss.write_index(self.faiss_index, self.faiss_index_path)
        
        # Save paper metadata
        papers_data = [
            {
                "title": p.title,
                "authors": p.authors,
                "abstract": p.abstract,
                "url": p.url,
                "source": p.source,
                "paper_id": p.paper_id
            }
            for p in self.papers
        ]
        
        with open(self.papers_metadata_path, 'w') as f:
            json.dump(papers_data, f, indent=2)
        
        print("✅ Index saved successfully!\n")
    
    def load_index(self):
        """Load FAISS index and paper metadata from disk"""
        print("📂 Loading FAISS index and metadata...")
        
        if os.path.exists(self.faiss_index_path) and os.path.exists(self.papers_metadata_path):
            # Load FAISS index
            self.faiss_index = faiss.read_index(self.faiss_index_path)
            
            # Load paper metadata
            with open(self.papers_metadata_path, 'r') as f:
                papers_data = json.load(f)
            
            self.papers = [
                Paper(
                    title=p["title"],
                    authors=p["authors"],
                    abstract=p["abstract"],
                    url=p["url"],
                    source=p["source"],
                    paper_id=p["paper_id"]
                )
                for p in papers_data
            ]
            
            print(f"✅ Loaded {len(self.papers)} papers from index\n")
        else:
            print("⚠️ No saved index found. Starting fresh.\n")


# Example usage demonstration
if __name__ == "__main__":
    # Initialize assistant
    assistant = ResearchAssistant(
        use_4bit_quantization=True  # Enable 4-bit quantization
    )
    
    # This would normally be populated from ArXiv/PubMed APIs (see api_connectors.py)
    # For demonstration, here's the structure:
    sample_papers = [
        Paper(
            title="Attention Is All You Need",
            authors=["Vaswani et al."],
            abstract="The dominant sequence transduction models are based on complex recurrent or convolutional neural networks...",
            url="https://arxiv.org/abs/1706.03762",
            source="arxiv",
            paper_id="1706.03762"
        )
    ]
    
    # Add papers to vector database
    assistant.add_papers(sample_papers)
    
    # Query the system
    result = assistant.query(
        "What are transformers in deep learning?",
        top_k=5
    )
    
    print("="*80)
    print("ANSWER:", result["answer"])
    print("\nCITATIONS:")
    for i, citation in enumerate(result["citations"], 1):
        print(f"{i}. {citation['title']} (Score: {citation['relevance_score']:.2f})")
