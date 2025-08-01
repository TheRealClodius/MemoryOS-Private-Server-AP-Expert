"""
Short-term memory management for MemoryOS
Redis-style in-memory storage with simple file persistence
"""

import json
import os
from collections import deque
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from .utils import get_timestamp, ensure_directory_exists


class ShortTermMemory:
    """
    Manages short-term memory storage and retrieval using Redis-style in-memory storage
    
    This implementation uses a deque for Redis-like performance:
    - Fast insertion and retrieval (O(1))
    - Automatic capacity management with FIFO eviction
    - Simple JSON persistence for recovery
    """
    
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
        # Use deque for Redis-like performance with maxlen for automatic eviction
        self.memory = deque(maxlen=capacity)
        self._load_memory()
    
    def _load_memory(self) -> None:
        """Load memory from file into deque"""
        try:
            with open(self.memory_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    self.memory = deque(data, maxlen=self.capacity)
                else:
                    self.memory = deque(maxlen=self.capacity)
                print(f"ShortTermMemory: Loaded {len(self.memory)} entries from {self.memory_file}")
        except FileNotFoundError:
            self.memory = deque(maxlen=self.capacity)
            print(f"ShortTermMemory: No history file found. Initializing new memory.")
        except json.JSONDecodeError:
            self.memory = deque(maxlen=self.capacity)
            print(f"ShortTermMemory: Error decoding JSON. Initializing new memory.")
        except Exception as e:
            self.memory = deque(maxlen=self.capacity)
            print(f"ShortTermMemory: Error loading memory: {e}. Initializing new memory.")
    
    def _save_memory(self) -> bool:
        """Save memory to file"""
        try:
            with open(self.memory_file, "w", encoding="utf-8") as f:
                json.dump(list(self.memory), f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            print(f"Error saving ShortTermMemory to {self.memory_file}: {e}")
            return False
    
    def add_qa_pair(
        self, 
        user_input: str, 
        agent_response: str, 
        message_id: Optional[str] = None,
        timestamp: Optional[str] = None,
        meta_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Add a question-answer pair to short-term memory
        
        Args:
            user_input: The user's input/question
            agent_response: The agent's response
            message_id: Optional message ID for linking conversation and execution memories
            timestamp: Optional timestamp (uses current time if not provided)
            meta_data: Optional metadata dictionary
            
        Returns:
            Dictionary containing the stored memory entry
        """
        if timestamp is None:
            timestamp = get_timestamp()
        
        # Generate message_id if not provided
        if message_id is None:
            import uuid
            message_id = str(uuid.uuid4())
        
        qa_pair = {
            "message_id": message_id,
            "user_input": user_input,
            "agent_response": agent_response,
            "timestamp": timestamp,
            "meta_data": meta_data or {},
            "access_count": 0,
            "last_accessed": timestamp
        }
        
        # Check if we need to handle overflow before adding
        evicted_entry = None
        if len(self.memory) >= self.capacity:
            # deque will automatically evict the oldest when we append
            # but we need to capture it first for overflow processing
            if len(self.memory) == self.capacity:
                evicted_entry = self.memory[0]  # Get the oldest entry
        
        # Add to memory (deque handles capacity automatically)
        self.memory.append(qa_pair)
        print(f"ShortTermMemory: Added QA. User: {user_input[:30]}...")
        
        # Handle overflow if an entry was evicted
        if evicted_entry:
            self._handle_overflow(evicted_entry)
        
        self._save_memory()
        return qa_pair
    
    def _handle_overflow(self, removed_entry: Dict[str, Any]) -> None:
        """
        Handle memory overflow by storing removed entries for consolidation
        
        Args:
            removed_entry: The memory entry that was removed due to capacity limit
        """
        overflow_file = os.path.join(self.data_path, "overflow.json")
        try:
            with open(overflow_file, "r", encoding="utf-8") as f:
                overflow_memory = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            overflow_memory = []
        
        overflow_memory.append(removed_entry)
        
        try:
            with open(overflow_file, "w", encoding="utf-8") as f:
                json.dump(overflow_memory, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Error saving overflow: {e}")
    
    def get_overflow_entries(self) -> List[Dict[str, Any]]:
        """Get all overflow entries for consolidation"""
        overflow_file = os.path.join(self.data_path, "overflow.json")
        try:
            with open(overflow_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return []
    
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
        return list(self.memory)
    
    def get_recent(self, count: int = 5) -> List[Dict[str, Any]]:
        """
        Get the most recent memory entries
        
        Args:
            count: Number of recent entries to return
            
        Returns:
            List of recent memory entries
        """
        if count <= 0:
            return []
        
        memory_list = list(self.memory)
        if count >= len(memory_list):
            return memory_list
        else:
            return memory_list[-count:]
    
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
        
        # Convert deque to list for easier processing
        memory_list = list(self.memory)
        
        for i, entry in enumerate(memory_list):
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
        
        # Convert deque to list for processing
        memory_list = list(self.memory)
        
        for entry in memory_list:
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
                "entries": list(self.memory)
            }
            try:
                with open(overflow_file, "w", encoding="utf-8") as f:
                    json.dump(backup_data, f, ensure_ascii=False, indent=2)
            except Exception as e:
                print(f"Error saving backup: {e}")
        
        self.memory.clear()
        return self._save_memory()
    
    def get_context_for_query(self, query: str, max_entries: int = 5) -> List[Dict[str, Any]]:
        """
        Get context from short-term memory - Redis-style fast access
        
        Original MemoryOS: Short-term memory provides recent context without semantic filtering.
        This is for immediate conversation flow, not deep semantic search.
        
        Args:
            query: The user query (kept for compatibility but not used for filtering)
            max_entries: Maximum number of entries to return
            
        Returns:
            List of recent memory entries (Redis-style FIFO access)
        """
        # Original MemoryOS approach: Return recent entries without complex filtering
        # Short-term memory is about recency, not semantic relevance
        return self.get_recent(max_entries)
    
    def export_for_consolidation(self) -> Dict[str, Any]:
        """Export data for mid-term memory consolidation"""
        return {
            "user_id": self.user_id,
            "export_timestamp": get_timestamp(),
            "current_memory": list(self.memory),
            "overflow_memory": self.get_overflow_entries(),
            "stats": self.get_memory_stats()
        }
    
    def pop_oldest(self) -> Optional[Dict[str, Any]]:
        """
        Pop the oldest entry from memory (Redis-style FIFO operation)
        
        Returns:
            The oldest memory entry if available, None otherwise
        """
        if self.memory:
            oldest_entry = self.memory.popleft()
            print("ShortTermMemory: Evicted oldest QA pair.")
            self._save_memory()
            return oldest_entry
        return None
    
    def is_empty(self) -> bool:
        """Check if short-term memory is empty"""
        return len(self.memory) == 0
