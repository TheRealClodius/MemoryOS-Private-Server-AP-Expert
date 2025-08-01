"""
Execution memory management for MemoryOS
Stores and retrieves execution details linked to conversation memories via message_id
"""

import json
import os
import numpy as np
from collections import deque
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from .utils import get_timestamp, ensure_directory_exists, safe_json_save, safe_json_load


class ExecutionMemory:
    """
    Manages execution memory storage and retrieval
    Stores execution details that can be linked to conversation memories via message_id
    """
    
    def __init__(self, user_id: str, data_path: str, capacity: int = 100):
        """
        Initialize execution memory
        
        Args:
            user_id: Unique identifier for the user
            data_path: Base path for data storage
            capacity: Maximum number of execution records to store
        """
        self.user_id = user_id
        self.capacity = capacity
        self.data_path = os.path.join(data_path, user_id, "execution_memory")
        ensure_directory_exists(self.data_path)
        
        self.memory_file = os.path.join(self.data_path, "executions.json")
        self.embeddings_file = os.path.join(self.data_path, "execution_embeddings.npy")
        
        # Use deque for automatic capacity management
        self.executions = deque(maxlen=capacity)
        self.embeddings: Optional[np.ndarray] = None
        
        self._load_memory()
        self._load_embeddings()
    
    def _load_memory(self) -> None:
        """Load execution memory from file"""
        try:
            data = safe_json_load(self.memory_file, [])
            if isinstance(data, list):
                self.executions = deque(data, maxlen=self.capacity)
                print(f"ExecutionMemory: Loaded {len(self.executions)} entries from {self.memory_file}")
            else:
                self.executions = deque(maxlen=self.capacity)
        except Exception as e:
            print(f"ExecutionMemory: Error loading memory: {e}. Initializing new memory.")
            self.executions = deque(maxlen=self.capacity)
    
    def _save_memory(self) -> bool:
        """Save execution memory to file"""
        try:
            return safe_json_save(list(self.executions), self.memory_file)
        except Exception as e:
            print(f"Error saving ExecutionMemory to {self.memory_file}: {e}")
            return False
    
    def _load_embeddings(self) -> None:
        """Load embeddings from file"""
        try:
            if os.path.exists(self.embeddings_file):
                self.embeddings = np.load(self.embeddings_file)
        except Exception as e:
            print(f"Error loading execution embeddings: {e}")
            self.embeddings = None
    
    def _save_embeddings(self) -> bool:
        """Save embeddings to file"""
        try:
            if self.embeddings is not None:
                np.save(self.embeddings_file, self.embeddings)
                return True
        except Exception as e:
            print(f"Error saving execution embeddings: {e}")
        return False
    
    def add_execution(
        self,
        message_id: str,
        execution_summary: str,
        tools_used: List[str],
        errors: List[Dict[str, str]],
        observations: str,
        success: bool,
        duration_ms: Optional[int] = None,
        timestamp: Optional[str] = None,
        meta_data: Optional[Dict[str, Any]] = None,
        embedding: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Add an execution record to memory
        
        Args:
            message_id: Message ID linking to conversation memory
            execution_summary: Summary of what was executed
            tools_used: List of tools used in chronological order
            errors: List of error dictionaries with error_type, error_message, tool
            observations: Observations about the execution
            success: Whether the execution was successful
            duration_ms: Execution duration in milliseconds
            timestamp: Optional timestamp
            meta_data: Optional metadata
            embedding: Optional embedding vector for the execution
            
        Returns:
            Dictionary containing the stored execution record
        """
        if timestamp is None:
            timestamp = get_timestamp()
        
        execution_record = {
            "message_id": message_id,
            "execution_summary": execution_summary,
            "tools_used": tools_used,
            "errors": errors,
            "observations": observations,
            "success": success,
            "duration_ms": duration_ms,
            "timestamp": timestamp,
            "meta_data": meta_data or {},
            "access_count": 0,
            "last_accessed": timestamp
        }
        
        # Check if we need to handle overflow
        evicted_entry = None
        if len(self.executions) >= self.capacity:
            if len(self.executions) == self.capacity:
                evicted_entry = self.executions[0]
        
        # Add to memory
        self.executions.append(execution_record)
        print(f"ExecutionMemory: Added execution for message_id: {message_id}")
        
        # Add embedding if provided
        if embedding is not None:
            if self.embeddings is None:
                self.embeddings = embedding.reshape(1, -1)
            else:
                self.embeddings = np.vstack([self.embeddings, embedding.reshape(1, -1)])
            
            # Handle embedding overflow if execution was evicted
            if evicted_entry and self.embeddings.shape[0] > self.capacity:
                self.embeddings = self.embeddings[1:]  # Remove oldest embedding
        
        self._save_memory()
        if embedding is not None:
            self._save_embeddings()
        
        return execution_record
    
    def get_execution_by_message_id(self, message_id: str) -> Optional[Dict[str, Any]]:
        """
        Get execution record by message_id
        
        Args:
            message_id: Message ID to search for
            
        Returns:
            Execution record if found, None otherwise
        """
        for execution in self.executions:
            if execution.get("message_id") == message_id:
                # Update access information
                execution["access_count"] = execution.get("access_count", 0) + 1
                execution["last_accessed"] = get_timestamp()
                self._save_memory()
                return execution
        return None
    
    def search_by_embedding(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        similarity_threshold: float = 0.3
    ) -> List[tuple[Dict[str, Any], float]]:
        """
        Search execution records by embedding similarity
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of top results to return
            similarity_threshold: Minimum similarity threshold
            
        Returns:
            List of (execution_record, similarity_score) tuples
        """
        if self.embeddings is None or len(self.executions) == 0:
            return []
        
        from .utils import compute_similarity
        similarities = []
        
        for i, execution_embedding in enumerate(self.embeddings):
            similarity = compute_similarity(query_embedding, execution_embedding)
            if similarity >= similarity_threshold:
                similarities.append((i, similarity))
        
        # Sort by similarity and get top k
        similarities.sort(key=lambda x: x[1], reverse=True)
        results = []
        
        for i, similarity in similarities[:top_k]:
            if i < len(self.executions):
                execution_record = list(self.executions)[i]
                # Update access information
                execution_record["access_count"] = execution_record.get("access_count", 0) + 1
                execution_record["last_accessed"] = get_timestamp()
                results.append((execution_record, similarity))
        
        if results:
            self._save_memory()
        
        return results
    
    def search_by_tools(self, tools: List[str]) -> List[Dict[str, Any]]:
        """
        Search execution records by tools used
        
        Args:
            tools: List of tools to search for
            
        Returns:
            List of matching execution records
        """
        results = []
        tool_set = set(tool.lower() for tool in tools)
        
        for execution in self.executions:
            execution_tools = set(tool.lower() for tool in execution.get("tools_used", []))
            if tool_set.intersection(execution_tools):
                execution["access_count"] = execution.get("access_count", 0) + 1
                execution["last_accessed"] = get_timestamp()
                results.append(execution)
        
        if results:
            self._save_memory()
        
        return results
    
    def search_by_success_status(self, success: bool) -> List[Dict[str, Any]]:
        """
        Search execution records by success status
        
        Args:
            success: Success status to filter by
            
        Returns:
            List of matching execution records
        """
        results = []
        for execution in self.executions:
            if execution.get("success") == success:
                results.append(execution)
        return results
    
    def get_execution_patterns(self) -> Dict[str, Any]:
        """
        Analyze execution patterns for learning
        
        Returns:
            Dictionary with execution pattern analysis
        """
        if not self.executions:
            return {
                "total_executions": 0,
                "success_rate": 0.0,
                "most_used_tools": [],
                "common_errors": [],
                "average_duration": 0.0
            }
        
        total_executions = len(self.executions)
        successful_executions = sum(1 for exec in self.executions if exec.get("success", False))
        success_rate = successful_executions / total_executions
        
        # Count tool usage
        tool_counts = {}
        error_counts = {}
        durations = []
        
        for execution in self.executions:
            # Count tools
            for tool in execution.get("tools_used", []):
                tool_counts[tool] = tool_counts.get(tool, 0) + 1
            
            # Count errors
            for error in execution.get("errors", []):
                error_type = error.get("error_type", "unknown")
                error_counts[error_type] = error_counts.get(error_type, 0) + 1
            
            # Collect durations
            duration = execution.get("duration_ms")
            if duration is not None:
                durations.append(duration)
        
        # Sort by frequency
        most_used_tools = sorted(tool_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        common_errors = sorted(error_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        
        average_duration = sum(durations) / len(durations) if durations else 0.0
        
        return {
            "total_executions": total_executions,
            "success_rate": success_rate,
            "most_used_tools": most_used_tools,
            "common_errors": common_errors,
            "average_duration": average_duration,
            "execution_count_by_success": {
                "successful": successful_executions,
                "failed": total_executions - successful_executions
            }
        }
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get statistics about execution memory"""
        return {
            "total_executions": len(self.executions),
            "capacity": self.capacity,
            "usage_percentage": (len(self.executions) / self.capacity) * 100,
            "has_embeddings": self.embeddings is not None,
            "embedding_count": self.embeddings.shape[0] if self.embeddings is not None else 0
        }
    
    def clear(self) -> bool:
        """Clear all execution memory"""
        # Backup before clearing
        if self.executions:
            backup_file = os.path.join(self.data_path, f"execution_backup_{get_timestamp().replace(':', '-')}.json")
            backup_data = {
                "cleared_at": get_timestamp(),
                "executions": list(self.executions)
            }
            safe_json_save(backup_data, backup_file)
        
        self.executions.clear()
        self.embeddings = None
        
        # Remove embedding file
        if os.path.exists(self.embeddings_file):
            os.remove(self.embeddings_file)
        
        return self._save_memory()