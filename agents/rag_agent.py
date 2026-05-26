import os
from pathlib import Path
import PyPDF2
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import torch

# ============================
# RAG AGENT - PDF LITERATURE REVIEW SEARCH
# ============================

class RAGAgent:
    def __init__(self, pdf_directory="data/literature_reviews", embedding_model="all-MiniLM-L6-v2"):
        """
        Initialize RAG agent with PDF directory and embedding model
        
        Args:
            pdf_directory: Path to folder containing PDF literature reviews
            embedding_model: Sentence-transformer model for embeddings
        """
        self.pdf_directory = pdf_directory
        self.embedding_model = SentenceTransformer(embedding_model)
        self.index = None
        self.documents = []
        self.doc_chunks = []
        
        print(f"[RAG Agent] Initializing with PDFs from: {pdf_directory}")
        self._load_and_index_pdfs()
    
    def _extract_text_from_pdf(self, pdf_path):
        """Extract text from PDF file"""
        try:
            text = ""
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                for page in reader.pages:
                    text += page.extract_text()
            return text
        except Exception as e:
            print(f"[RAG Agent] Error reading {pdf_path}: {e}")
            return ""
    
    def _chunk_text(self, text, chunk_size=500, overlap=50):
        """Split text into overlapping chunks"""
        chunks = []
        words = text.split()
        chunk_words = []
        
        for word in words:
            chunk_words.append(word)
            if len(chunk_words) >= chunk_size:
                chunks.append(" ".join(chunk_words))
                chunk_words = chunk_words[-(overlap):]
        
        if chunk_words:
            chunks.append(" ".join(chunk_words))
        
        return chunks
    
    def _load_and_index_pdfs(self):
        """Load all PDFs from directory and create FAISS index"""
        if not os.path.exists(self.pdf_directory):
            print(f"[RAG Agent] Directory not found: {self.pdf_directory}")
            return
        
        pdf_files = list(Path(self.pdf_directory).glob("*.pdf"))
        
        if not pdf_files:
            print(f"[RAG Agent] No PDF files found in {self.pdf_directory}")
            return
        
        print(f"[RAG Agent] Found {len(pdf_files)} PDF(s)")
        
        all_chunks = []
        all_embeddings = []
        
        for pdf_file in pdf_files:
            print(f"[RAG Agent] Processing: {pdf_file.name}")
            
            # Extract text
            text = self._extract_text_from_pdf(str(pdf_file))
            if not text:
                continue
            
            # Chunk text
            chunks = self._chunk_text(text)
            
            for chunk in chunks:
                if chunk.strip():  # Skip empty chunks
                    all_chunks.append({
                        "source": pdf_file.name,
                        "content": chunk
                    })
        
        if not all_chunks:
            print("[RAG Agent] No text extracted from PDFs")
            return
        
        print(f"[RAG Agent] Created {len(all_chunks)} text chunks")
        
        # Create embeddings
        print("[RAG Agent] Generating embeddings... (this may take a moment)")
        chunk_texts = [chunk["content"] for chunk in all_chunks]
        embeddings = self.embedding_model.encode(chunk_texts, convert_to_numpy=True)
        
        # Create FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings.astype(np.float32))
        
        self.documents = all_chunks
        print(f"[RAG Agent] FAISS index created with {len(all_chunks)} documents")
    
    def search_relevant_literature(self, query, top_k=5):
        """
        Search for relevant literature based on query
        
        Args:
            query: Query string (e.g., summarizer output insights)
            top_k: Number of top results to return
            
        Returns:
            List of relevant document chunks with sources
        """
        if self.index is None or not self.documents:
            print("[RAG Agent] No documents indexed. Cannot search.")
            return []
        
        # Encode query
        query_embedding = self.embedding_model.encode([query], convert_to_numpy=True)
        
        # Search
        distances, indices = self.index.search(query_embedding.astype(np.float32), top_k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.documents):
                doc = self.documents[idx]
                results.append({
                    "source": doc["source"],
                    "content": doc["content"][:300] + "...",  # Truncate for display
                    "relevance_score": float(1 / (1 + distances[0][i]))  # Convert distance to relevance
                })
        
        return results
    
    def generate_intelligence_report(self, summarizer_output):
        """
        Generate intelligence report by cross-referencing summarizer output with literature
        
        Args:
            summarizer_output: Dict with commodity analysis from summarizer agent
            
        Returns:
            Dict with intelligence findings
        """
        intelligence = {}
        
        for commodity, analysis in summarizer_output.items():
            print(f"\n[RAG Agent] Finding literature for {commodity}...")
            
            # Create search query from sentiment and drivers
            query_parts = [
                analysis.get("sentiment", ""),
                " ".join(analysis.get("drivers", [])),
                analysis.get("summary", "")
            ]
            search_query = " ".join(filter(None, query_parts))
            
            # Search literature
            relevant_docs = self.search_relevant_literature(search_query, top_k=3)
            
            intelligence[commodity] = {
                "market_analysis": analysis,
                "literature_review_findings": relevant_docs,
                "has_supporting_evidence": len(relevant_docs) > 0
            }
        
        return intelligence


# ============================
# MAIN RAG EXECUTION
# ============================

def run_rag_agent(summarizer_output, pdf_directory="data/literature_reviews"):
    """
    Run RAG agent to find relevant literature
    
    Args:
        summarizer_output: Output from summarizer_agent
        pdf_directory: Path to PDF literature reviews
        
    Returns:
        Intelligence report with literature cross-references
    """
    rag = RAGAgent(pdf_directory=pdf_directory)
    intelligence_report = rag.generate_intelligence_report(summarizer_output)
    return intelligence_report


if __name__ == "__main__":
    # Test function (placeholder)
    print("RAG Agent loaded successfully")
