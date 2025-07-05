"""
Short-term memory management for MemoryOS
"""

import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from .utils import get_timestamp, safe_json_save, safe_json_load, ensure_directory_exists


class ShortTermMemory:
    """Manages short-term memory storage and retrieval"""
    
    def __init__(self, user_id: str, data_path: str, capacity: int = 10):
        """
        Initialize short-term memory
        
        Args:
            user_id: Unique identifier for the user
            data_path: Base path for data storage
            capacity: Maximum number of QA pairs to store
        """
        self.user_id = user_id
        self.capacity = capacity
        self.data_path = os.path.join(data_path, user_id, "short_term")
        ensure_directory_exists(self.data_path)
        
        self.memory_file = os.path.join(self.data_path, "memory.json")
        self.memory: List[Dict[str, Any]] = self._load_memory()
    
    def _load_memory(self) -> List[Dict[str, Any]]:
        """Load memory from file"""
        return safe_json_load(self.memory_file, [])
    
    def _save_memory(self) -> bool:
        """Save memory to file"""
        return safe_json_save(self.memory, self.memory_file)
    
    def add_qa_pair(
        self, 
        user_input: str, 
        agent_response: str, 
        timestamp: Optional[str] = None,
        meta_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Add a question-answer pair to short-term memory
        
        Args:
            user_input: The user's input/question
            agent_response: The agent's response
            timestamp: Optional timestamp (uses current time if not provided)
            meta_data: Optional metadata dictionary
            
        Returns:
            Dictionary containing the stored memory entry
        """
        if timestamp is None:
            timestamp = get_timestamp()
        
        qa_pair = {
            "user_input": user_input,
            "agent_response": agent_response,
            "timestamp": timestamp,
            "meta_data": meta_data or {},
            "access_count": 0,
            "last_accessed": timestamp
        }
        
        # Add to memory
        self.memory.append(qa_pair)
        
        # Maintain capacity limit
        if len(self.memory) > self.capacity:
            removed = self.memory.pop(0)  # Remove oldest
            self._handle_overflow(removed)
        
        self._save_memory()
        return qa_pair
    
    def _handle_overflow(self, removed_entry: Dict[str, Any]) -> None:
        """
        Handle memory overflow by storing removed entries for consolidation
        
        Args:
            removed_entry: The memory entry that was removed due to capacity limit
        """
        overflow_file = os.path.join(self.data_path, "overflow.json")
        overflow_memory = safe_json_load(overflow_file, [])
        overflow_memory.append(removed_entry)
        safe_json_save(overflow_memory, overflow_file)
    
    def get_overflow_entries(self) -> List[Dict[str, Any]]:
        """Get all overflow entries for consolidation"""
        overflow_file = os.path.join(self.data_path, "overflow.json")
        return safe_json_load(overflow_file, [])
    
    def clear_overflow(self) -> bool:
        """Clear overflow entries after consolidation"""
        overflow_file = os.path.join(self.data_path, "overflow.json")
        try:
            if os.path.exists(overflow_file):
                os.remove(overflow_file)
            return True
        except Exception as e:
            print(f"Error clearing overflow: {e}")
            return False
    
    def get_all(self) -> List[Dict[str, Any]]:
        """Get all short-term memory entries"""
        return self.memory.copy()
    
    def get_recent(self, count: int = 5) -> List[Dict[str, Any]]:
        """
        Get the most recent memory entries
        
        Args:
            count: Number of recent entries to return
            
        Returns:
            List of recent memory entries
        """
        return self.memory[-count:] if count <= len(self.memory) else self.memory.copy()
    
    def search_by_keyword(self, keyword: str) -> List[Dict[str, Any]]:
        """
        Search memory entries by keyword
        
        Args:
            keyword: Keyword to search for
            
        Returns:
            List of matching memory entries
        """
        keyword_lower = keyword.lower()
        results = []
        
        for entry in self.memory:
            user_input = entry.get("user_input", "").lower()
            agent_response = entry.get("agent_response", "").lower()
            
            if keyword_lower in user_input or keyword_lower in agent_response:
                # Update access count
                entry["access_count"] = entry.get("access_count", 0) + 1
                entry["last_accessed"] = get_timestamp()
                results.append(entry)
        
        if results:
            self._save_memory()
        
        return results
    
    def get_by_timeframe(self, hours: int = 24) -> List[Dict[str, Any]]:
        """
        Get memory entries within a specific timeframe
        
        Args:
            hours: Number of hours to look back
            
        Returns:
            List of memory entries within the timeframe
        """
        cutoff_time = datetime.now() - timedelta(hours=hours)
        results = []
        
        for entry in self.memory:
            try:
                entry_time = datetime.fromisoformat(entry["timestamp"].replace('Z', '+00:00'))
                if entry_time >= cutoff_time:
                    results.append(entry)
            except (ValueError, KeyError):
                # Skip entries with invalid timestamps
                continue
        
        return results
    
    def update_access(self, entry_index: int) -> bool:
        """
        Update access information for a memory entry
        
        Args:
            entry_index: Index of the entry to update
            
        Returns:
            True if update was successful, False otherwise
        """
        if 0 <= entry_index < len(self.memory):
            entry = self.memory[entry_index]
            entry["access_count"] = entry.get("access_count", 0) + 1
            entry["last_accessed"] = get_timestamp()
            return self._save_memory()
        return False
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get statistics about short-term memory"""
        if not self.memory:
            return {
                "total_entries": 0,
                "capacity": self.capacity,
                "usage_percentage": 0.0,
                "oldest_entry": None,
                "newest_entry": None,
                "most_accessed": None
            }
        
        # Find most accessed entry
        most_accessed = max(self.memory, key=lambda x: x.get("access_count", 0))
        
        return {
            "total_entries": len(self.memory),
            "capacity": self.capacity,
            "usage_percentage": (len(self.memory) / self.capacity) * 100,
            "oldest_entry": self.memory[0]["timestamp"] if self.memory else None,
            "newest_entry": self.memory[-1]["timestamp"] if self.memory else None,
            "most_accessed": {
                "entry": most_accessed,
                "access_count": most_accessed.get("access_count", 0)
            }
        }
    
    def is_full(self) -> bool:
        """Check if short-term memory is at capacity"""
        return len(self.memory) >= self.capacity
    
    def clear(self) -> bool:
        """Clear all short-term memory"""
        # Store cleared entries in overflow for potential recovery
        if self.memory:
            overflow_file = os.path.join(self.data_path, "cleared_backup.json")
            backup_data = {
                "cleared_at": get_timestamp(),
                "entries": self.memory.copy()
            }
            safe_json_save(backup_data, overflow_file)
        
        self.memory = []
        return self._save_memory()
    
    def get_context_for_query(self, query: str, max_entries: int = 5) -> List[Dict[str, Any]]:
        """
        Get relevant context from short-term memory for a query
        
        Args:
            query: The user query
            max_entries: Maximum number of entries to return
            
        Returns:
            List of relevant memory entries
        """
        # Simple relevance scoring based on keyword matching
        query_words = set(query.lower().split())
        scored_entries = []
        
        for entry in self.memory:
            score = 0
            
            # Score based on keyword matches
            user_input_words = set(entry.get("user_input", "").lower().split())
            agent_response_words = set(entry.get("agent_response", "").lower().split())
            
            # Calculate overlap
            user_overlap = len(query_words.intersection(user_input_words))
            agent_overlap = len(query_words.intersection(agent_response_words))
            
            score = user_overlap * 2 + agent_overlap  # Weight user input higher
            
            if score > 0:
                scored_entries.append((score, entry))
        
        # Sort by score and return top entries
        scored_entries.sort(key=lambda x: x[0], reverse=True)
        return [entry for _, entry in scored_entries[:max_entries]]
    
    def export_for_consolidation(self) -> Dict[str, Any]:
        """Export data for mid-term memory consolidation"""
        return {
            "user_id": self.user_id,
            "export_timestamp": get_timestamp(),
            "current_memory": self.memory.copy(),
            "overflow_memory": self.get_overflow_entries(),
            "stats": self.get_memory_stats()
        }
