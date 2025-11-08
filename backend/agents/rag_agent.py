# agents/rag_agent.py - RAG (Retrieval-Augmented Generation) Agent

from typing import Dict, Any, Optional, List, Tuple
from agents.base import BaseAgent, AgentResponse, AgentType
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
import openai
import re

class RAGAgent(BaseAgent):
    """
    RAG Agent - Retrieves and synthesizes information from indexed documents
    Uses Pinecone for vector search and generates responses with citations
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("RAG", AgentType.RAG, config)
        
        # Initialize embedding model
        embed_model_name = config.get("embed_model", "sentence-transformers/all-MiniLM-L6-v2")
        self.embed_model = SentenceTransformer(embed_model_name)
        
        # Initialize Pinecone
        if config and "pinecone_config" in config:
            pc_cfg = config["pinecone_config"]
            self.pinecone_client = Pinecone(api_key=pc_cfg["api_key"])
            self.index = self.pinecone_client.Index(pc_cfg["index_name"])
        else:
            self.index = None
        
        # LLM configuration (optional - for synthesis)
        self.use_llm = config.get("use_llm", False)
        if self.use_llm and "openai_api_key" in config:
            openai.api_key = config["openai_api_key"]
    
    async def execute(self, input_data: Dict[str, Any], context: Optional[Dict] = None) -> AgentResponse:
        """
        Retrieve relevant documents and synthesize response
        """
        try:
            query = input_data.get("query", "")
            top_k = input_data.get("top_k", 5)
            filters = input_data.get("filters", {})
            synthesis_required = input_data.get("synthesis", True)
            
            # Retrieve relevant documents
            retrieved_docs = await self._retrieve_documents(query, top_k, filters)
            
            if not retrieved_docs:
                return AgentResponse(
                    success=True,
                    data={
                        "documents": [],
                        "answer": "No relevant documents found for your query."
                    },
                    message="No documents retrieved"
                )
            
            # Generate citations
            citations = self._generate_citations(retrieved_docs)
            
            # Synthesize answer if required
            if synthesis_required:
                answer = await self._synthesize_answer(query, retrieved_docs, context)
            else:
                answer = self._extract_relevant_passages(query, retrieved_docs)
            
            return AgentResponse(
                success=True,
                data={
                    "documents": retrieved_docs,
                    "answer": answer,
                    "citations": citations
                },
                message=f"Retrieved {len(retrieved_docs)} relevant documents",
                citations=citations,
                metadata={"retrieval_score": self._calculate_relevance_score(retrieved_docs)}
            )
            
        except Exception as e:
            return AgentResponse(
                success=False,
                data=None,
                message=f"RAG retrieval failed: {str(e)}"
            )
    
    async def _retrieve_documents(self, query: str, top_k: int, filters: Dict) -> List[Dict]:
        """Retrieve relevant documents from Pinecone"""
        if not self.index:
            return []
        
        # Generate query embedding
        query_embedding = self.embed_model.encode(query).tolist()
        
        # Build Pinecone filters
        pinecone_filters = self._build_pinecone_filters(filters)
        
        # Query Pinecone
        results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True,
            filter=pinecone_filters if pinecone_filters else None
        )
        
        # Format results
        documents = []
        for match in results['matches']:
            doc = {
                "id": match['id'],
                "score": match['score'],
                "metadata": match.get('metadata', {}),
                "text": self._extract_text_from_metadata(match.get('metadata', {}))
            }
            documents.append(doc)
        
        return documents
    
    def _build_pinecone_filters(self, filters: Dict) -> Optional[Dict]:
        """Build Pinecone filter query from input filters"""
        if not filters:
            return None
        
        pinecone_filter = {}
        
        if "location" in filters:
            pinecone_filter["location"] = {"$eq": filters["location"]}
        
        if "price_min" in filters and "price_max" in filters:
            pinecone_filter["price"] = {
                "$gte": filters["price_min"],
                "$lte": filters["price_max"]
            }
        elif "price_max" in filters:
            pinecone_filter["price"] = {"$lte": filters["price_max"]}
        
        if "property_type" in filters:
            pinecone_filter["property_type"] = {"$eq": filters["property_type"]}
        
        return pinecone_filter if pinecone_filter else None
    
    def _extract_text_from_metadata(self, metadata: Dict) -> str:
        """Extract relevant text from document metadata"""
        text_parts = []
        
        # Extract main text fields
        for field in ["title", "long_description", "certs_text"]:
            if field in metadata and metadata[field]:
                text_parts.append(str(metadata[field]))
        
        # Extract location and price info
        if "location" in metadata:
            text_parts.append(f"Location: {metadata['location']}")
        
        if "price" in metadata:
            text_parts.append(f"Price: {metadata['price']}")
        
        return " ".join(text_parts)
    
    async def _synthesize_answer(self, query: str, documents: List[Dict], context: Optional[Dict]) -> str:
        """Synthesize answer from retrieved documents"""
        
        if self.use_llm:
            # Use LLM for synthesis
            return await self._llm_synthesis(query, documents, context)
        else:
            # Use rule-based synthesis
            return self._rule_based_synthesis(query, documents)
    
    async def _llm_synthesis(self, query: str, documents: List[Dict], context: Optional[Dict]) -> str:
        """Use LLM to synthesize answer from documents"""
        # Prepare context from documents
        doc_context = "\n\n".join([
            f"Document {i+1} (Score: {doc['score']:.2f}):\n{doc['text'][:500]}"
            for i, doc in enumerate(documents[:3])  # Use top 3 documents
        ])
        
        # Create prompt
        prompt = f"""Based on the following documents, answer the query concisely and accurately.
        
Query: {query}

Context from documents:
{doc_context}

Provide a comprehensive answer based on the information in the documents. If the documents don't contain relevant information, say so."""

        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful real estate assistant that provides accurate information based on provided documents."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=300,
                temperature=0.3
            )
            
            return response.choices[0].message.content
        except Exception as e:
            self.logger.error(f"LLM synthesis failed: {str(e)}")
            # Fallback to rule-based
            return self._rule_based_synthesis(query, documents)
    
    def _rule_based_synthesis(self, query: str, documents: List[Dict]) -> str:
        """Rule-based synthesis of answer from documents"""
        if not documents:
            return "No relevant information found."
        
        # Extract key information from top documents
        key_info = []
        
        for i, doc in enumerate(documents[:3]):
            metadata = doc.get("metadata", {})
            
            # Extract property information
            if "title" in metadata:
                info = f"Property: {metadata['title']}"
                if "location" in metadata:
                    info += f" in {metadata['location']}"
                if "price" in metadata:
                    info += f" (₹{metadata['price']:,})"
                key_info.append(info)
            
            # Extract relevant snippets
            text = doc.get("text", "")
            query_terms = query.lower().split()
            
            # Find sentences containing query terms
            sentences = text.split(". ")
            relevant_sentences = []
            for sentence in sentences:
                sentence_lower = sentence.lower()
                if any(term in sentence_lower for term in query_terms):
                    relevant_sentences.append(sentence)
            
            if relevant_sentences:
                key_info.append(". ".join(relevant_sentences[:2]))
        
        # Compile answer
        answer = "Based on the available documents:\n\n"
        answer += "\n\n".join(key_info)
        
        return answer
    
    def _extract_relevant_passages(self, query: str, documents: List[Dict]) -> str:
        """Extract relevant passages without synthesis"""
        passages = []
        query_terms = set(query.lower().split())
        
        for doc in documents[:3]:
            text = doc.get("text", "")
            sentences = text.split(". ")
            
            # Score sentences based on query term overlap
            scored_sentences = []
            for sentence in sentences:
                sentence_lower = sentence.lower()
                score = sum(1 for term in query_terms if term in sentence_lower)
                if score > 0:
                    scored_sentences.append((score, sentence))
            
            # Sort by score and take top sentences
            scored_sentences.sort(key=lambda x: x[0], reverse=True)
            top_sentences = [s[1] for s in scored_sentences[:2]]
            
            if top_sentences:
                passage = ". ".join(top_sentences)
                passages.append(f"[Document {doc['id']}]: {passage}")
        
        return "\n\n".join(passages)
    
    def _generate_citations(self, documents: List[Dict]) -> List[Dict]:
        """Generate proper citations for retrieved documents"""
        citations = []
        
        for i, doc in enumerate(documents):
            metadata = doc.get("metadata", {})
            
            citation = {
                "index": i + 1,
                "source_id": doc["id"],
                "title": metadata.get("title", "Untitled"),
                "location": metadata.get("location", ""),
                "relevance_score": doc["score"],
                "url": metadata.get("image_file", "")  # Link to property image/details
            }
            
            # Create citation text
            citation_text = f"[{i+1}] {citation['title']}"
            if citation['location']:
                citation_text += f", {citation['location']}"
            
            citation["text"] = citation_text
            citations.append(citation)
        
        return citations
    
    def _calculate_relevance_score(self, documents: List[Dict]) -> float:
        """Calculate overall relevance score of retrieved documents"""
        if not documents:
            return 0.0
        
        # Average of top 3 document scores
        top_scores = [doc["score"] for doc in documents[:3]]
        return sum(top_scores) / len(top_scores)

    async def search_similar_properties(self, property_features: Dict, top_k: int = 5) -> List[Dict]:
        """Search for similar properties based on features"""
        # Create feature text for embedding
        feature_text = f"{property_features.get('location', '')} {property_features.get('bhk', '')} BHK {property_features.get('property_type', '')}"
        
        # Retrieve similar properties
        similar = await self._retrieve_documents(feature_text, top_k, {})
        
        return similar