"""
Vector-based Retrieval-Augmented Generation (RAG) tool.

This module implements a proper vector-based RAG system using sentence-transformers
for embeddings and ChromaDB for vector storage, as specified in the project requirements.
It provides domain-specific RAG tools for different business areas.
"""

from __future__ import annotations

import os
import hashlib
from typing import Dict, List, Optional, Any
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
    import chromadb
    from chromadb.config import Settings
    HAS_VECTOR_DEPS = True
except ImportError:
    HAS_VECTOR_DEPS = False
    print("Warning: Vector dependencies not installed. Install with: pip install sentence-transformers chromadb")

import database


class VectorRAGTool:
    """Vector-based RAG tool using sentence-transformers and ChromaDB."""
    
    def __init__(self, collection_name: str = "documents"):
        self.collection_name = collection_name
        self.model = None
        self.chroma_client = None
        self.collection = None
        
        if HAS_VECTOR_DEPS:
            self._initialize_vector_components()
        else:
            print("Warning: Vector RAG not available. Falling back to basic search.")
    
    def _initialize_vector_components(self):
        """Initialize the sentence transformer model and ChromaDB client."""
        try:
            # Initialize sentence transformer model
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Initialize ChromaDB client
            self.chroma_client = chromadb.PersistentClient(
                path="./vector_db",
                settings=Settings(anonymized_telemetry=False)
            )
            
            # Get or create collection
            self.collection = self.chroma_client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            
            # Load documents if collection is empty
            self._load_documents_if_needed()
            
        except Exception as e:
            print(f"Warning: Failed to initialize vector components: {e}")
            self.model = None
            self.chroma_client = None
            self.collection = None
    
    def _load_documents_if_needed(self):
        """Load documents into vector database if not already present."""
        if not self.collection:
            return
            
        # Check if we need to reload documents
        try:
            count = self.collection.count()
            if count > 0:
                return  # Documents already loaded
        except:
            pass
        
        print("Loading documents into vector database...")
        self._load_all_documents()
    
    def _load_all_documents(self):
        """Load all documents from the database into the vector store."""
        if not self.model or not self.collection:
            return
            
        rows = database.query("SELECT id, module, path, tags FROM documents", None)
        
        documents = []
        metadatas = []
        ids = []
        
        for row in rows:
            doc_id = row["id"]
            module = row["module"]
            path = row["path"]
            tags = row["tags"] or ""
            
            # Read document content
            text = ""
            if path and os.path.isfile(path):
                try:
                    with open(path, "r", encoding="utf-8", errors="ignore") as f:
                        text = f.read()
                except Exception as e:
                    print(f"Error reading {path}: {e}")
                    continue
            
            if not text.strip():
                continue
            
            # Split document into chunks for better retrieval
            chunks = self._split_document(text, path)
            
            for i, chunk in enumerate(chunks):
                chunk_id = f"{doc_id}_{i}"
                documents.append(chunk)
                metadatas.append({
                    "doc_id": doc_id,
                    "module": module,
                    "path": path,
                    "tags": tags,
                    "chunk_index": i
                })
                ids.append(chunk_id)
        
        if documents:
            # Add documents to collection in batches
            batch_size = 100
            for i in range(0, len(documents), batch_size):
                batch_docs = documents[i:i+batch_size]
                batch_metas = metadatas[i:i+batch_size]
                batch_ids = ids[i:i+batch_size]
                
                try:
                    self.collection.add(
                        documents=batch_docs,
                        metadatas=batch_metas,
                        ids=batch_ids
                    )
                except Exception as e:
                    print(f"Error adding batch {i//batch_size}: {e}")
            
            print(f"Loaded {len(documents)} document chunks into vector database")
    
    def _split_document(self, text: str, path: str) -> List[str]:
        """Split document into smaller chunks for better retrieval."""
        # Simple chunking strategy - split by paragraphs and limit size
        paragraphs = text.split('\n\n')
        chunks = []
        current_chunk = ""
        max_chunk_size = 500  # characters
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
                
            if len(current_chunk) + len(paragraph) > max_chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = paragraph
            else:
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph
        
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        # Ensure we have at least one chunk
        if not chunks and text.strip():
            chunks = [text[:max_chunk_size]]
        
        return chunks
    
    def search(
        self,
        query: str,
        k: int = 3,
        module: Optional[str] = None,
        tags: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search for relevant documents using vector similarity.
        
        Args:
            query: Natural language query string.
            k: Maximum number of results to return.
            module: Optional filter to restrict results to a given module.
            tags: Optional comma-separated list of tags to filter documents.
        
        Returns:
            A list of dictionaries with document information and excerpts.
        """
        if not self.model or not self.collection:
            # Fallback to simple search if vector components not available
            return self._fallback_search(query, k, module, tags)
        
        try:
            # Build filter conditions
            where_conditions = {}
            if module:
                where_conditions["module"] = module
            
            if tags:
                # For tags, we'll do a post-processing filter since ChromaDB
                # doesn't support complex string matching in where clause
                pass
            
            # Query the vector database
            results = self.collection.query(
                query_texts=[query],
                n_results=min(k * 2, 20),  # Get more results for filtering
                where=where_conditions if where_conditions else None
            )
            
            if not results["documents"] or not results["documents"][0]:
                return []
            
            # Process results
            processed_results = []
            for i, (doc, metadata, distance) in enumerate(zip(
                results["documents"][0],
                results["metadatas"][0], 
                results["distances"][0]
            )):
                # Apply tag filtering if specified
                if tags:
                    requested_tags = {t.strip().lower() for t in tags.split(",") if t.strip()}
                    doc_tags = {t.strip().lower() for t in (metadata.get("tags", "") or "").split(",") if t.strip()}
                    if not requested_tags.issubset(doc_tags):
                        continue
                
                # Convert distance to similarity score (0-1, higher is better)
                score = max(0, 1 - distance)
                
                processed_results.append({
                    "id": metadata["doc_id"],
                    "module": metadata["module"],
                    "tags": metadata["tags"],
                    "score": score,
                    "excerpt": doc[:200] + "..." if len(doc) > 200 else doc,
                    "path": metadata["path"]
                })
                
                if len(processed_results) >= k:
                    break
            
            return processed_results
            
        except Exception as e:
            print(f"Error in vector search: {e}")
            return self._fallback_search(query, k, module, tags)
    
    def _fallback_search(self, query: str, k: int, module: Optional[str], tags: Optional[str]) -> List[Dict[str, Any]]:
        """Fallback to simple text search when vector search is not available."""
        # Simple keyword matching fallback
        query_words = set(query.lower().split())
        rows = database.query("SELECT id, module, path, tags FROM documents", None)
        
        results = []
        for row in rows:
            if module and row["module"] != module:
                continue
            
            if tags:
                requested_tags = {t.strip().lower() for t in tags.split(",") if t.strip()}
                doc_tags = {t.strip().lower() for t in (row["tags"] or "").split(",") if t.strip()}
                if not requested_tags.issubset(doc_tags):
                    continue
            
            # Read document and calculate simple score
            path = row["path"]
            text = ""
            if path and os.path.isfile(path):
                try:
                    with open(path, "r", encoding="utf-8", errors="ignore") as f:
                        text = f.read()
                except:
                    continue
            
            if not text:
                continue
            
            # Simple word overlap score
            doc_words = set(text.lower().split())
            overlap = len(query_words & doc_words)
            total = len(query_words | doc_words)
            score = overlap / total if total > 0 else 0
            
            if score > 0:
                results.append({
                    "id": row["id"],
                    "module": row["module"], 
                    "tags": row["tags"],
                    "score": score,
                    "excerpt": text[:200] + "..." if len(text) > 200 else text,
                    "path": row["path"]
                })
        
        # Sort by score and return top k
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:k]


# Domain-specific RAG tools as required by project specifications
class PolicyRAGTool(VectorRAGTool):
    """Finance policy document search tool."""
    
    def __init__(self):
        super().__init__("policy_documents")
    
    def search_policies(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        """Search specifically in policy documents."""
        return self.search(query, k=k, tags="policy")


class DocRAGTool(VectorRAGTool):
    """General document search tool for supplier contracts and incident reports."""
    
    def __init__(self):
        super().__init__("business_documents")
    
    def search_contracts(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        """Search supplier contracts and procurement documents."""
        return self.search(query, k=k, module="inventory", tags="contracts")
    
    def search_incidents(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        """Search incident reports."""
        return self.search(query, k=k, tags="incidents")


class DefinitionRAGTool(VectorRAGTool):
    """Business glossary and metric definition search tool."""
    
    def __init__(self):
        super().__init__("definition_documents")
    
    def search_definitions(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        """Search business definitions and glossary."""
        return self.search(query, k=k, tags="glossary,metrics")


class SalesRAGTool(VectorRAGTool):
    """Sales and CRM document search tool."""
    
    def __init__(self):
        super().__init__("sales_documents")
    
    def search_procedures(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        """Search sales procedures and CRM documentation."""
        return self.search(query, k=k, module="sales")
