"""
Memory retrieval system for MemoryOS
"""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from .utils import compute_similarity, get_timestamp


class MemoryRetriever:
    """Handles retrieval of relevant memories across all memory layers"""
    
    def __init__(self, short_term_memory, mid_term_memory, long_term_memory, queue_capacity: int = 7):
        """
        Initialize memory retriever
        
        Args:
            short_term_memory: ShortTermMemory instance
            mid_term_memory: MidTermMemory instance
            long_term_memory: LongTermMemory instance
            queue_capacity: Maximum number of items to return for each memory type
        """
        self.short_term_memory = short_term_memory
        self.mid_term_memory = mid_term_memory
        self.long_term_memory = long_term_memory
        self.queue_capacity = queue_capacity
    
    def retrieve_context(self, user_query: str, user_id: str, 
                        query_embedding: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Retrieve comprehensive context for a user query
        
        Args:
            user_query: The user's query
            user_id: User identifier
            query_embedding: Pre-computed query embedding (optional)
            
        Returns:
            Dictionary containing retrieved memories from all layers
        """
        context = {
            "query": user_query,
            "user_id": user_id,
            "timestamp": get_timestamp(),
            "short_term_memory": [],
            "retrieved_pages": [],
            "retrieved_user_knowledge": [],
            "retrieved_assistant_knowledge": []
        }
        
        # Get short-term memory context
        context["short_term_memory"] = self._get_short_term_context(user_query)
        
        # Get mid-term memory context (if embedding is available)
        if query_embedding is not None:
            context["retrieved_pages"] = self._get_mid_term_context(query_embedding)
            
            # Get long-term knowledge context
            user_knowledge = self._get_user_knowledge_context(query_embedding)
            assistant_knowledge = self._get_assistant_knowledge_context(query_embedding)
            
            context["retrieved_user_knowledge"] = user_knowledge
            context["retrieved_assistant_knowledge"] = assistant_knowledge
        
        return context
    
    def _get_short_term_context(self, user_query: str) -> List[Dict[str, Any]]:
        """Get recent context from short-term memory - Redis-style fast access without filtering"""
        try:
            # Original MemoryOS: Simply return recent memory entries without similarity filtering
            # Short-term memory is for immediate context, not semantic search
            recent_memory = self.short_term_memory.get_recent(self.queue_capacity)
            
            # Add basic relevance score for compatibility but don't filter
            for entry in recent_memory:
                # Simple keyword presence check for score (but don't filter out)
                query_words = set(user_query.lower().split())
                user_input_words = set(entry.get("user_input", "").lower().split())
                agent_response_words = set(entry.get("agent_response", "").lower().split())
                
                user_overlap = len(query_words.intersection(user_input_words))
                agent_overlap = len(query_words.intersection(agent_response_words))
                
                # Score for ordering but don't exclude any entries
                score = user_overlap * 2 + agent_overlap
                entry["similarity_score"] = score if score > 0 else 1.0  # Default score for recency
            
            # Sort by score but include all recent entries
            recent_memory.sort(key=lambda x: x.get("similarity_score", 1.0), reverse=True)
            return recent_memory
        
        except Exception as e:
            print(f"Error retrieving short-term context: {e}")
            return []
    
    def _get_mid_term_context(self, query_embedding: np.ndarray) -> List[Dict[str, Any]]:
        """Get relevant context from mid-term memory with proper similarity filtering"""
        try:
            # Search by embedding similarity with higher threshold
            results = self.mid_term_memory.search_by_embedding(
                query_embedding, 
                top_k=self.queue_capacity,
                similarity_threshold=0.7  # Increased from 0.3 to 0.7 for relevance
            )
            
            # Format results for output
            formatted_results = []
            for segment, similarity in results:
                # Extract QA pairs and create formatted entries
                for qa_pair in segment.get("qa_pairs", []):
                    formatted_entry = {
                        "user_input": qa_pair.get("user_input", ""),
                        "agent_response": qa_pair.get("agent_response", ""),
                        "timestamp": qa_pair.get("timestamp", segment.get("timestamp", "")),
                        "meta_info": {
                            "segment_id": segment.get("id"),
                            "similarity_score": similarity,
                            "segment_summary": segment.get("summary", ""),
                            "themes": segment.get("themes", []),
                            "heat": segment.get("heat", 0.0)
                        }
                    }
                    formatted_results.append(formatted_entry)
            
            return formatted_results[:self.queue_capacity]
        
        except Exception as e:
            print(f"Error retrieving mid-term context: {e}")
            return []
    
    def _get_user_knowledge_context(self, query_embedding: np.ndarray) -> List[Dict[str, Any]]:
        """Get relevant user knowledge with proper similarity filtering"""
        try:
            results = self.long_term_memory.search_user_knowledge(
                query_embedding,
                top_k=self.queue_capacity,
                similarity_threshold=0.7  # Increased from 0.3 to 0.7 for relevance
            )
            
            formatted_results = []
            for knowledge_entry, similarity in results:
                formatted_entry = {
                    "knowledge": knowledge_entry.get("knowledge", ""),
                    "timestamp": knowledge_entry.get("timestamp", ""),
                    "source": knowledge_entry.get("source", ""),
                    "confidence": knowledge_entry.get("confidence", 1.0),
                    "similarity_score": similarity
                }
                formatted_results.append(formatted_entry)
            
            return formatted_results
        
        except Exception as e:
            print(f"Error retrieving user knowledge: {e}")
            return []
    
    def _get_assistant_knowledge_context(self, query_embedding: np.ndarray) -> List[Dict[str, Any]]:
        """Get relevant assistant knowledge with proper similarity filtering"""
        try:
            results = self.long_term_memory.search_assistant_knowledge(
                query_embedding,
                top_k=self.queue_capacity,
                similarity_threshold=0.7  # Increased from 0.3 to 0.7 for relevance
            )
            
            formatted_results = []
            for knowledge_entry, similarity in results:
                formatted_entry = {
                    "knowledge": knowledge_entry.get("knowledge", ""),
                    "timestamp": knowledge_entry.get("timestamp", ""),
                    "source": knowledge_entry.get("source", ""),
                    "confidence": knowledge_entry.get("confidence", 1.0),
                    "similarity_score": similarity
                }
                formatted_results.append(formatted_entry)
            
            return formatted_results
        
        except Exception as e:
            print(f"Error retrieving assistant knowledge: {e}")
            return []
    
    def search_by_timeframe(self, hours: int = 24) -> Dict[str, List[Dict[str, Any]]]:
        """
        Search memories within a specific timeframe
        
        Args:
            hours: Number of hours to look back
            
        Returns:
            Dictionary with memories from each layer within timeframe
        """
        results = {
            "short_term": [],
            "mid_term": [],
            "user_knowledge": [],
            "assistant_knowledge": []
        }
        
        try:
            # Short-term memory
            results["short_term"] = self.short_term_memory.get_by_timeframe(hours)
            
            # Mid-term memory
            results["mid_term"] = self.mid_term_memory.get_recent_segments(hours)
            
            # Long-term memory (filter by timestamp)
            from datetime import datetime, timedelta
            cutoff_time = datetime.now() - timedelta(hours=hours)
            
            for knowledge_entry in self.long_term_memory.get_user_knowledge():
                try:
                    entry_time = datetime.fromisoformat(knowledge_entry["timestamp"].replace('Z', '+00:00'))
                    if entry_time >= cutoff_time:
                        results["user_knowledge"].append(knowledge_entry)
                except (ValueError, KeyError):
                    continue
            
            for knowledge_entry in self.long_term_memory.get_assistant_knowledge():
                try:
                    entry_time = datetime.fromisoformat(knowledge_entry["timestamp"].replace('Z', '+00:00'))
                    if entry_time >= cutoff_time:
                        results["assistant_knowledge"].append(knowledge_entry)
                except (ValueError, KeyError):
                    continue
        
        except Exception as e:
            print(f"Error in timeframe search: {e}")
        
        return results
    
    def search_by_keywords(self, keywords: List[str]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Search memories by keywords across all layers
        
        Args:
            keywords: List of keywords to search for
            
        Returns:
            Dictionary with matching memories from each layer
        """
        results = {
            "short_term": [],
            "mid_term": [],
            "user_knowledge": [],
            "assistant_knowledge": []
        }
        
        try:
            # Search short-term memory
            for keyword in keywords:
                matches = self.short_term_memory.search_by_keyword(keyword)
                results["short_term"].extend(matches)
            
            # Remove duplicates from short-term results
            seen_timestamps = set()
            unique_short_term = []
            for entry in results["short_term"]:
                timestamp = entry.get("timestamp")
                if timestamp not in seen_timestamps:
                    seen_timestamps.add(timestamp)
                    unique_short_term.append(entry)
            results["short_term"] = unique_short_term
            
            # Search mid-term memory by themes and content
            keyword_set = set(kw.lower() for kw in keywords)
            for segment in self.mid_term_memory.memory_segments:
                segment_themes = set(theme.lower() for theme in segment.get("themes", []))
                summary_words = set(segment.get("summary", "").lower().split())
                
                # Check for keyword matches in themes or summary
                if keyword_set.intersection(segment_themes) or keyword_set.intersection(summary_words):
                    results["mid_term"].append(segment)
            
            # Search long-term knowledge
            for knowledge_entry in self.long_term_memory.get_user_knowledge():
                knowledge_text = knowledge_entry.get("knowledge", "").lower()
                if any(keyword.lower() in knowledge_text for keyword in keywords):
                    results["user_knowledge"].append(knowledge_entry)
            
            for knowledge_entry in self.long_term_memory.get_assistant_knowledge():
                knowledge_text = knowledge_entry.get("knowledge", "").lower()
                if any(keyword.lower() in knowledge_text for keyword in keywords):
                    results["assistant_knowledge"].append(knowledge_entry)
        
        except Exception as e:
            print(f"Error in keyword search: {e}")
        
        return results
    
    def get_memory_summary(self) -> Dict[str, Any]:
        """Get a summary of all memory layers"""
        try:
            short_term_stats = self.short_term_memory.get_memory_stats()
            mid_term_stats = self.mid_term_memory.get_memory_stats()
            long_term_stats = self.long_term_memory.get_memory_stats()
            
            return {
                "timestamp": get_timestamp(),
                "short_term": short_term_stats,
                "mid_term": mid_term_stats,
                "long_term": long_term_stats,
                "total_memories": {
                    "short_term_count": short_term_stats.get("total_entries", 0),
                    "mid_term_count": mid_term_stats.get("total_segments", 0),
                    "user_knowledge_count": long_term_stats.get("user_knowledge", {}).get("total_entries", 0),
                    "assistant_knowledge_count": long_term_stats.get("assistant_knowledge", {}).get("total_entries", 0)
                }
            }
        
        except Exception as e:
            print(f"Error getting memory summary: {e}")
            return {"error": str(e), "timestamp": get_timestamp()}
    
    def find_related_memories(self, memory_id: str, memory_type: str) -> List[Dict[str, Any]]:
        """
        Find memories related to a specific memory entry
        
        Args:
            memory_id: ID of the memory to find relations for
            memory_type: Type of memory ("short_term", "mid_term", "user_knowledge", "assistant_knowledge")
            
        Returns:
            List of related memories
        """
        related_memories = []
        
        try:
            # This is a simplified implementation
            # In a full implementation, you might use embeddings to find semantic similarities
            
            if memory_type == "mid_term":
                # Find the segment and look for similar themes
                target_segment = None
                for segment in self.mid_term_memory.memory_segments:
                    if segment.get("id") == memory_id:
                        target_segment = segment
                        break
                
                if target_segment:
                    target_themes = set(target_segment.get("themes", []))
                    for segment in self.mid_term_memory.memory_segments:
                        if segment.get("id") != memory_id:
                            segment_themes = set(segment.get("themes", []))
                            if target_themes.intersection(segment_themes):
                                related_memories.append(segment)
            
            # Similar logic could be implemented for other memory types
            
        except Exception as e:
            print(f"Error finding related memories: {e}")
        
        return related_memories
    
    def get_user_interaction_patterns(self) -> Dict[str, Any]:
        """Analyze user interaction patterns across memory layers"""
        try:
            patterns = {
                "most_frequent_topics": [],
                "interaction_times": [],
                "communication_style": "neutral",
                "engagement_level": "medium"
            }
            
            # Analyze themes from mid-term memory
            theme_counts = {}
            for segment in self.mid_term_memory.memory_segments:
                for theme in segment.get("themes", []):
                    theme_counts[theme] = theme_counts.get(theme, 0) + 1
            
            # Sort themes by frequency
            sorted_themes = sorted(theme_counts.items(), key=lambda x: x[1], reverse=True)
            patterns["most_frequent_topics"] = [theme for theme, count in sorted_themes[:10]]
            
            # Analyze interaction times from short-term memory
            interactions = self.short_term_memory.get_all()
            patterns["interaction_times"] = [entry.get("timestamp") for entry in interactions]
            
            # Simple engagement analysis based on response lengths
            if interactions:
                avg_user_length = np.mean([len(entry.get("user_input", "")) for entry in interactions])
                avg_response_length = np.mean([len(entry.get("agent_response", "")) for entry in interactions])
                
                if avg_user_length > 100 and avg_response_length > 200:
                    patterns["engagement_level"] = "high"
                elif avg_user_length < 50 or avg_response_length < 100:
                    patterns["engagement_level"] = "low"
            
            return patterns
        
        except Exception as e:
            print(f"Error analyzing interaction patterns: {e}")
            return {"error": str(e)}
