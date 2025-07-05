"""
Main MemoryOS class - orchestrates all memory components
"""

import os
import numpy as np
from typing import Dict, List, Optional, Any
import openai

from .short_term import ShortTermMemory
from .mid_term import MidTermMemory
from .long_term import LongTermMemory
from .retriever import MemoryRetriever
from .updater import MemoryUpdater
from .utils import get_timestamp, ensure_directory_exists
from .prompts import RESPONSE_GENERATION_PROMPT


class Memoryos:
    """Main MemoryOS class for personalized AI agent memory management"""
    
    def __init__(
        self,
        user_id: str,
        openai_api_key: str,
        data_storage_path: str = "./memoryos_data",
        openai_base_url: Optional[str] = None,
        assistant_id: str = "default_assistant",
        short_term_capacity: int = 10,
        mid_term_capacity: int = 2000,
        long_term_knowledge_capacity: int = 100,
        retrieval_queue_capacity: int = 7,
        mid_term_heat_threshold: float = 5.0,
        llm_model: str = "gpt-4o-mini",
        embedding_model: str = "text-embedding-3-small"
    ):
        """
        Initialize MemoryOS
        
        Args:
            user_id: Unique identifier for the user
            openai_api_key: OpenAI API key
            data_storage_path: Path for storing memory data
            openai_base_url: Optional custom OpenAI base URL
            assistant_id: Unique identifier for the assistant
            short_term_capacity: Maximum short-term memory entries
            mid_term_capacity: Maximum mid-term memory segments
            long_term_knowledge_capacity: Maximum knowledge entries
            retrieval_queue_capacity: Maximum retrieval results per category
            mid_term_heat_threshold: Heat threshold for long-term promotion
            llm_model: LLM model for text processing
            embedding_model: Model for generating embeddings
        """
        self.user_id = user_id
        self.assistant_id = assistant_id
        self.data_storage_path = data_storage_path
        self.llm_model = llm_model
        self.embedding_model = embedding_model
        
        # Setup OpenAI client
        self.openai_client = openai.OpenAI(
            api_key=openai_api_key,
            base_url=openai_base_url
        )
        
        # Ensure data directory exists
        ensure_directory_exists(os.path.join(data_storage_path, user_id))
        
        # Initialize memory components
        self.short_term_memory = ShortTermMemory(
            user_id=user_id,
            data_path=data_storage_path,
            capacity=short_term_capacity
        )
        
        self.mid_term_memory = MidTermMemory(
            user_id=user_id,
            data_path=data_storage_path,
            capacity=mid_term_capacity,
            heat_threshold=mid_term_heat_threshold
        )
        
        self.user_long_term_memory = LongTermMemory(
            user_id=user_id,
            assistant_id=assistant_id,
            data_path=data_storage_path,
            knowledge_capacity=long_term_knowledge_capacity
        )
        
        self.retriever = MemoryRetriever(
            short_term_memory=self.short_term_memory,
            mid_term_memory=self.mid_term_memory,
            long_term_memory=self.user_long_term_memory,
            queue_capacity=retrieval_queue_capacity
        )
        
        self.updater = MemoryUpdater(
            short_term_memory=self.short_term_memory,
            mid_term_memory=self.mid_term_memory,
            long_term_memory=self.user_long_term_memory,
            embedding_function=self._generate_embedding,
            llm_function=self._call_llm
        )
    
    def _generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for text using OpenAI API"""
        try:
            response = self.openai_client.embeddings.create(
                input=text,
                model=self.embedding_model
            )
            embedding = np.array(response.data[0].embedding, dtype=np.float32)
            return embedding
        except Exception as e:
            print(f"Error generating embedding: {e}")
            # Return zero vector as fallback
            return np.zeros(1536, dtype=np.float32)  # text-embedding-3-small dimension
    
    def _call_llm(self, prompt: str, max_tokens: int = 1000) -> str:
        """Call LLM for text processing"""
        try:
            response = self.openai_client.chat.completions.create(
                model=self.llm_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0.7
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error calling LLM: {e}")
            return ""
    
    def add_memory(
        self,
        user_input: str,
        agent_response: str,
        timestamp: Optional[str] = None,
        meta_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Add a new memory (user input and agent response pair)
        
        Args:
            user_input: The user's input
            agent_response: The agent's response
            timestamp: Optional timestamp
            meta_data: Optional metadata
            
        Returns:
            Dictionary with operation result
        """
        try:
            # Add to short-term memory
            qa_pair = self.short_term_memory.add_qa_pair(
                user_input=user_input,
                agent_response=agent_response,
                timestamp=timestamp,
                meta_data=meta_data
            )
            
            # Process overflow if short-term memory is full
            if self.short_term_memory.is_full():
                self.updater.process_short_term_overflow()
            
            # Process hot segments periodically
            hot_segments = self.mid_term_memory.get_hot_segments()
            if hot_segments:
                self.updater.process_hot_segments()
            
            return {
                "status": "success",
                "message": "Memory added successfully",
                "qa_pair": qa_pair
            }
        
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error adding memory: {str(e)}"
            }
    
    def get_response(
        self,
        query: str,
        relationship_with_user: str = "assistant",
        style_hint: str = ""
    ) -> str:
        """
        Generate a response using comprehensive memory context
        
        Args:
            query: User query
            relationship_with_user: Relationship context
            style_hint: Style hint for response
            
        Returns:
            Generated response
        """
        try:
            # Generate query embedding
            query_embedding = self._generate_embedding(query)
            
            # Retrieve context from all memory layers
            context = self.retriever.retrieve_context(
                user_query=query,
                user_id=self.user_id,
                query_embedding=query_embedding
            )
            
            # Get user profile
            user_profile = self.get_user_profile_summary()
            
            # Format context for prompt
            short_term_context = self._format_short_term_context(context["short_term_memory"])
            mid_term_context = self._format_mid_term_context(context["retrieved_pages"])
            user_knowledge_context = self._format_knowledge_context(context["retrieved_user_knowledge"])
            assistant_knowledge_context = self._format_knowledge_context(context["retrieved_assistant_knowledge"])
            
            # Generate response using LLM
            response_prompt = RESPONSE_GENERATION_PROMPT.format(
                query=query,
                user_profile=user_profile,
                short_term_memory=short_term_context,
                mid_term_memory=mid_term_context,
                user_knowledge=user_knowledge_context,
                assistant_knowledge=assistant_knowledge_context
            )
            
            response = self._call_llm(response_prompt, max_tokens=1500)
            
            return response
        
        except Exception as e:
            return f"I apologize, but I encountered an error while processing your request: {str(e)}"
    
    def _format_short_term_context(self, short_term_entries: List[Dict[str, Any]]) -> str:
        """Format short-term memory for prompt"""
        if not short_term_entries:
            return "No recent conversation history."
        
        formatted = ""
        for entry in short_term_entries[-5:]:  # Last 5 entries
            formatted += f"User: {entry.get('user_input', '')}\n"
            formatted += f"Assistant: {entry.get('agent_response', '')}\n\n"
        
        return formatted.strip()
    
    def _format_mid_term_context(self, mid_term_entries: List[Dict[str, Any]]) -> str:
        """Format mid-term memory for prompt"""
        if not mid_term_entries:
            return "No relevant past interactions found."
        
        formatted = ""
        for entry in mid_term_entries:
            formatted += f"Past Interaction:\n"
            formatted += f"User: {entry.get('user_input', '')}\n"
            formatted += f"Assistant: {entry.get('agent_response', '')}\n"
            meta_info = entry.get('meta_info', {})
            if meta_info.get('segment_summary'):
                formatted += f"Context: {meta_info['segment_summary']}\n"
            formatted += "\n"
        
        return formatted.strip()
    
    def _format_knowledge_context(self, knowledge_entries: List[Dict[str, Any]]) -> str:
        """Format knowledge entries for prompt"""
        if not knowledge_entries:
            return "No relevant knowledge found."
        
        formatted = ""
        for entry in knowledge_entries:
            formatted += f"- {entry.get('knowledge', '')}\n"
        
        return formatted.strip()
    
    def get_user_profile_summary(self) -> str:
        """Get user profile summary"""
        return self.user_long_term_memory.get_user_profile_summary()
    
    def get_assistant_knowledge_summary(self) -> List[Dict[str, Any]]:
        """Get assistant knowledge summary"""
        return self.user_long_term_memory.get_assistant_knowledge()
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get comprehensive memory statistics"""
        try:
            return {
                "user_id": self.user_id,
                "assistant_id": self.assistant_id,
                "timestamp": get_timestamp(),
                "short_term": self.short_term_memory.get_memory_stats(),
                "mid_term": self.mid_term_memory.get_memory_stats(),
                "long_term": self.user_long_term_memory.get_memory_stats(),
                "update_stats": self.updater.get_update_statistics()
            }
        except Exception as e:
            return {"error": str(e), "timestamp": get_timestamp()}
    
    def search_memories(
        self,
        query: str,
        search_type: str = "embedding",
        max_results: int = 10
    ) -> Dict[str, Any]:
        """
        Search memories across all layers
        
        Args:
            query: Search query
            search_type: Type of search ("embedding", "keyword", "timeframe")
            max_results: Maximum results to return
            
        Returns:
            Search results from all memory layers
        """
        try:
            if search_type == "embedding":
                query_embedding = self._generate_embedding(query)
                return self.retriever.retrieve_context(
                    user_query=query,
                    user_id=self.user_id,
                    query_embedding=query_embedding
                )
            elif search_type == "keyword":
                keywords = query.split()
                return self.retriever.search_by_keywords(keywords)
            elif search_type == "timeframe":
                # Extract hours from query (simple implementation)
                hours = 24  # Default
                try:
                    # Look for numbers in query that might indicate hours
                    import re
                    numbers = re.findall(r'\d+', query)
                    if numbers:
                        hours = int(numbers[0])
                except:
                    pass
                return self.retriever.search_by_timeframe(hours)
            else:
                return {"error": f"Unknown search type: {search_type}"}
        
        except Exception as e:
            return {"error": str(e), "timestamp": get_timestamp()}
    
    def export_memory_data(self) -> Dict[str, Any]:
        """Export all memory data"""
        try:
            return {
                "user_id": self.user_id,
                "assistant_id": self.assistant_id,
                "export_timestamp": get_timestamp(),
                "short_term_data": self.short_term_memory.export_for_consolidation(),
                "mid_term_segments": self.mid_term_memory.memory_segments,
                "long_term_data": self.user_long_term_memory.export_user_data(),
                "memory_stats": self.get_memory_stats()
            }
        except Exception as e:
            return {"error": str(e), "timestamp": get_timestamp()}
    
    def clear_all_memory(self, backup: bool = True) -> bool:
        """
        Clear all memory data
        
        Args:
            backup: Whether to create backup before clearing
            
        Returns:
            True if successful
        """
        try:
            if backup:
                backup_data = self.export_memory_data()
                backup_file = os.path.join(
                    self.data_storage_path, 
                    self.user_id, 
                    f"full_backup_{get_timestamp().replace(':', '-')}.json"
                )
                ensure_directory_exists(os.path.dirname(backup_file))
                
                import json
                with open(backup_file, 'w', encoding='utf-8') as f:
                    json.dump(backup_data, f, indent=2, ensure_ascii=False)
            
            # Clear all memory layers
            self.short_term_memory.clear()
            self.mid_term_memory.memory_segments = []
            self.mid_term_memory.embeddings = None
            self.mid_term_memory._save_memory()
            self.mid_term_memory._save_embeddings()
            self.user_long_term_memory.clear_user_data(backup=False)
            
            return True
        
        except Exception as e:
            print(f"Error clearing memory: {e}")
            return False
    
    def consolidate_memories(self) -> Dict[str, Any]:
        """Manually trigger memory consolidation"""
        try:
            # Process any pending short-term overflow
            overflow_processed = self.updater.process_short_term_overflow()
            
            # Process hot segments
            hot_segments_processed = self.updater.process_hot_segments()
            
            # Consolidate similar memories
            consolidation_stats = self.updater.consolidate_similar_memories()
            
            return {
                "status": "success",
                "overflow_processed": overflow_processed,
                "hot_segments_processed": hot_segments_processed,
                "consolidation_stats": consolidation_stats,
                "timestamp": get_timestamp()
            }
        
        except Exception as e:
            return {
                "status": "error",
                "message": str(e),
                "timestamp": get_timestamp()
            }
