"""
Long-term memory management for MemoryOS
"""

import json
import os
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from .utils import (
    get_timestamp, safe_json_save, safe_json_load, ensure_directory_exists,
    compute_similarity, generate_hash
)


class LongTermMemory:
    """Manages long-term memory including user profile and knowledge bases"""
    
    def __init__(self, user_id: str, assistant_id: str, data_path: str, knowledge_capacity: int = 100):
        """
        Initialize long-term memory
        
        Args:
            user_id: Unique identifier for the user
            assistant_id: Unique identifier for the assistant
            data_path: Base path for data storage
            knowledge_capacity: Maximum number of knowledge entries
        """
        self.user_id = user_id
        self.assistant_id = assistant_id
        self.knowledge_capacity = knowledge_capacity
        self.data_path = os.path.join(data_path, user_id, "long_term")
        ensure_directory_exists(self.data_path)
        
        # File paths
        self.user_profile_file = os.path.join(self.data_path, "user_profile.json")
        self.user_knowledge_file = os.path.join(self.data_path, "user_knowledge.json")
        self.assistant_knowledge_file = os.path.join(self.data_path, f"assistant_knowledge_{assistant_id}.json")
        self.user_embeddings_file = os.path.join(self.data_path, "user_knowledge_embeddings.npy")
        self.assistant_embeddings_file = os.path.join(self.data_path, f"assistant_knowledge_embeddings_{assistant_id}.npy")
        
        # Load data
        self.user_profile: Dict[str, Any] = self._load_user_profile()
        self.user_knowledge: List[Dict[str, Any]] = self._load_user_knowledge()
        self.assistant_knowledge: List[Dict[str, Any]] = self._load_assistant_knowledge()
        self.user_embeddings: Optional[np.ndarray] = self._load_embeddings(self.user_embeddings_file)
        self.assistant_embeddings: Optional[np.ndarray] = self._load_embeddings(self.assistant_embeddings_file)
    
    def _load_user_profile(self) -> Dict[str, Any]:
        """Load user profile from file"""
        default_profile = {
            "user_id": self.user_id,
            "created_at": get_timestamp(),
            "last_updated": get_timestamp(),
            "profile_text": "None",
            "personality_traits": [],
            "interests": [],
            "preferences": {},
            "goals": [],
            "communication_style": "neutral",
            "update_count": 0
        }
        return safe_json_load(self.user_profile_file, default_profile)
    
    def _save_user_profile(self) -> bool:
        """Save user profile to file"""
        self.user_profile["last_updated"] = get_timestamp()
        return safe_json_save(self.user_profile, self.user_profile_file)
    
    def _load_user_knowledge(self) -> List[Dict[str, Any]]:
        """Load user knowledge from file"""
        return safe_json_load(self.user_knowledge_file, [])
    
    def _save_user_knowledge(self) -> bool:
        """Save user knowledge to file"""
        return safe_json_save(self.user_knowledge, self.user_knowledge_file)
    
    def _load_assistant_knowledge(self) -> List[Dict[str, Any]]:
        """Load assistant knowledge from file"""
        return safe_json_load(self.assistant_knowledge_file, [])
    
    def _save_assistant_knowledge(self) -> bool:
        """Save assistant knowledge to file"""
        return safe_json_save(self.assistant_knowledge, self.assistant_knowledge_file)
    
    def _load_embeddings(self, file_path: str) -> Optional[np.ndarray]:
        """Load embeddings from file"""
        try:
            if os.path.exists(file_path):
                return np.load(file_path)
        except Exception as e:
            print(f"Error loading embeddings from {file_path}: {e}")
        return None
    
    def _save_embeddings(self, embeddings: np.ndarray, file_path: str) -> bool:
        """Save embeddings to file"""
        try:
            if embeddings is not None:
                np.save(file_path, embeddings)
                return True
        except Exception as e:
            print(f"Error saving embeddings to {file_path}: {e}")
        return False
    
    def update_user_profile(self, new_profile_text: str, extracted_insights: Dict[str, Any] = None) -> bool:
        """
        Update user profile with new information
        
        Args:
            new_profile_text: Updated profile text
            extracted_insights: Additional insights extracted from conversations
            
        Returns:
            True if update was successful
        """
        # Update profile text
        self.user_profile["profile_text"] = new_profile_text
        self.user_profile["update_count"] = self.user_profile.get("update_count", 0) + 1
        
        # Update insights if provided
        if extracted_insights:
            for key, value in extracted_insights.items():
                if key in self.user_profile and isinstance(self.user_profile[key], list):
                    # Merge lists while avoiding duplicates
                    if isinstance(value, list):
                        existing_set = set(self.user_profile[key])
                        new_items = [item for item in value if item not in existing_set]
                        self.user_profile[key].extend(new_items)
                    else:
                        if value not in self.user_profile[key]:
                            self.user_profile[key].append(value)
                elif key in self.user_profile and isinstance(self.user_profile[key], dict):
                    # Merge dictionaries
                    if isinstance(value, dict):
                        self.user_profile[key].update(value)
                else:
                    # Direct assignment for other types
                    self.user_profile[key] = value
        
        return self._save_user_profile()
    
    def add_user_knowledge(self, knowledge: str, embedding: np.ndarray, 
                          source: str = "conversation", confidence: float = 1.0) -> bool:
        """
        Add knowledge about the user
        
        Args:
            knowledge: Knowledge text
            embedding: Embedding vector for the knowledge
            source: Source of the knowledge
            confidence: Confidence score for the knowledge
            
        Returns:
            True if knowledge was added successfully
        """
        knowledge_entry = {
            "id": generate_hash(f"{self.user_id}_{knowledge}_{get_timestamp()}"),
            "knowledge": knowledge,
            "timestamp": get_timestamp(),
            "source": source,
            "confidence": confidence,
            "access_count": 0,
            "last_accessed": get_timestamp()
        }
        
        # Check for duplicates
        if self._is_duplicate_knowledge(knowledge, self.user_knowledge):
            return False
        
        # Add knowledge
        self.user_knowledge.append(knowledge_entry)
        
        # Add embedding
        if self.user_embeddings is None:
            self.user_embeddings = embedding.reshape(1, -1)
        else:
            self.user_embeddings = np.vstack([self.user_embeddings, embedding.reshape(1, -1)])
        
        # Maintain capacity
        if len(self.user_knowledge) > self.knowledge_capacity:
            self._handle_user_knowledge_overflow()
        
        self._save_user_knowledge()
        self._save_embeddings(self.user_embeddings, self.user_embeddings_file)
        return True
    
    def add_assistant_knowledge(self, knowledge: str, embedding: np.ndarray,
                               source: str = "interaction", confidence: float = 1.0) -> bool:
        """
        Add knowledge for the assistant
        
        Args:
            knowledge: Knowledge text
            embedding: Embedding vector for the knowledge
            source: Source of the knowledge
            confidence: Confidence score for the knowledge
            
        Returns:
            True if knowledge was added successfully
        """
        knowledge_entry = {
            "id": generate_hash(f"{self.assistant_id}_{knowledge}_{get_timestamp()}"),
            "knowledge": knowledge,
            "timestamp": get_timestamp(),
            "source": source,
            "confidence": confidence,
            "access_count": 0,
            "last_accessed": get_timestamp()
        }
        
        # Check for duplicates
        if self._is_duplicate_knowledge(knowledge, self.assistant_knowledge):
            return False
        
        # Add knowledge
        self.assistant_knowledge.append(knowledge_entry)
        
        # Add embedding
        if self.assistant_embeddings is None:
            self.assistant_embeddings = embedding.reshape(1, -1)
        else:
            self.assistant_embeddings = np.vstack([self.assistant_embeddings, embedding.reshape(1, -1)])
        
        # Maintain capacity
        if len(self.assistant_knowledge) > self.knowledge_capacity:
            self._handle_assistant_knowledge_overflow()
        
        self._save_assistant_knowledge()
        self._save_embeddings(self.assistant_embeddings, self.assistant_embeddings_file)
        return True
    
    def _is_duplicate_knowledge(self, new_knowledge: str, knowledge_list: List[Dict[str, Any]]) -> bool:
        """Check if knowledge already exists (simple text matching)"""
        new_knowledge_lower = new_knowledge.lower().strip()
        for entry in knowledge_list:
            existing_knowledge = entry.get("knowledge", "").lower().strip()
            if new_knowledge_lower == existing_knowledge:
                return True
            # Check for high similarity (simple word overlap)
            new_words = set(new_knowledge_lower.split())
            existing_words = set(existing_knowledge.split())
            if len(new_words) > 0 and len(existing_words) > 0:
                overlap = len(new_words.intersection(existing_words))
                similarity = overlap / max(len(new_words), len(existing_words))
                if similarity > 0.8:  # High similarity threshold
                    return True
        return False
    
    def _handle_user_knowledge_overflow(self) -> None:
        """Handle user knowledge capacity overflow"""
        self._handle_knowledge_overflow(
            self.user_knowledge, 
            self.user_embeddings, 
            self.user_embeddings_file,
            "user_knowledge_archived.json"
        )
    
    def _handle_assistant_knowledge_overflow(self) -> None:
        """Handle assistant knowledge capacity overflow"""
        self._handle_knowledge_overflow(
            self.assistant_knowledge,
            self.assistant_embeddings,
            self.assistant_embeddings_file,
            f"assistant_knowledge_archived_{self.assistant_id}.json"
        )
    
    def _handle_knowledge_overflow(self, knowledge_list: List[Dict[str, Any]], 
                                 embeddings: Optional[np.ndarray], embeddings_file: str,
                                 archive_file: str) -> None:
        """Handle knowledge overflow by removing least accessed entries"""
        if len(knowledge_list) <= self.knowledge_capacity:
            return
        
        # Sort by access count and confidence
        knowledge_with_scores = []
        for i, entry in enumerate(knowledge_list):
            score = entry.get("access_count", 0) * entry.get("confidence", 1.0)
            knowledge_with_scores.append((score, i, entry))
        
        knowledge_with_scores.sort(key=lambda x: x[0])  # Sort by score ascending
        
        # Remove lowest scoring entries
        to_remove = len(knowledge_list) - self.knowledge_capacity
        removed_indices = []
        archived_entries = []
        
        for i in range(to_remove):
            _, idx, entry = knowledge_with_scores[i]
            removed_indices.append(idx)
            entry["archived_at"] = get_timestamp()
            archived_entries.append(entry)
        
        # Archive removed entries
        archive_path = os.path.join(self.data_path, archive_file)
        existing_archive = safe_json_load(archive_path, [])
        existing_archive.extend(archived_entries)
        safe_json_save(existing_archive, archive_path)
        
        # Remove from knowledge list and embeddings
        removed_indices.sort(reverse=True)
        for idx in removed_indices:
            knowledge_list.pop(idx)
            if embeddings is not None and idx < len(embeddings):
                embeddings = np.delete(embeddings, idx, axis=0)
        
        # Update embeddings reference
        if embeddings_file == self.user_embeddings_file:
            self.user_embeddings = embeddings
        else:
            self.assistant_embeddings = embeddings
    
    def search_user_knowledge(self, query_embedding: np.ndarray, top_k: int = 5,
                             similarity_threshold: float = 0.3) -> List[Tuple[Dict[str, Any], float]]:
        """
        Search user knowledge by embedding similarity
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of top results to return
            similarity_threshold: Minimum similarity threshold
            
        Returns:
            List of (knowledge_entry, similarity_score) tuples
        """
        return self._search_knowledge(
            self.user_knowledge, self.user_embeddings, 
            query_embedding, top_k, similarity_threshold
        )
    
    def search_assistant_knowledge(self, query_embedding: np.ndarray, top_k: int = 5,
                                  similarity_threshold: float = 0.3) -> List[Tuple[Dict[str, Any], float]]:
        """
        Search assistant knowledge by embedding similarity
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of top results to return
            similarity_threshold: Minimum similarity threshold
            
        Returns:
            List of (knowledge_entry, similarity_score) tuples
        """
        return self._search_knowledge(
            self.assistant_knowledge, self.assistant_embeddings,
            query_embedding, top_k, similarity_threshold
        )
    
    def _search_knowledge(self, knowledge_list: List[Dict[str, Any]], embeddings: Optional[np.ndarray],
                         query_embedding: np.ndarray, top_k: int, 
                         similarity_threshold: float) -> List[Tuple[Dict[str, Any], float]]:
        """Generic knowledge search method"""
        if embeddings is None or len(knowledge_list) == 0:
            return []
        
        similarities = []
        for i, knowledge_embedding in enumerate(embeddings):
            similarity = compute_similarity(query_embedding, knowledge_embedding)
            if similarity >= similarity_threshold:
                similarities.append((i, similarity))
        
        # Sort by similarity and get top k
        similarities.sort(key=lambda x: x[1], reverse=True)
        results = []
        
        for i, similarity in similarities[:top_k]:
            if i < len(knowledge_list):
                knowledge_entry = knowledge_list[i]
                # Update access information
                knowledge_entry["access_count"] = knowledge_entry.get("access_count", 0) + 1
                knowledge_entry["last_accessed"] = get_timestamp()
                results.append((knowledge_entry, similarity))
        
        return results
    
    def get_user_profile_summary(self) -> str:
        """Get user profile summary text"""
        return self.user_profile.get("profile_text", "None")
    
    def get_user_knowledge(self) -> List[Dict[str, Any]]:
        """Get all user knowledge entries"""
        return self.user_knowledge.copy()
    
    def get_assistant_knowledge(self) -> List[Dict[str, Any]]:
        """Get all assistant knowledge entries"""
        return self.assistant_knowledge.copy()
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get statistics about long-term memory"""
        return {
            "user_profile": {
                "last_updated": self.user_profile.get("last_updated"),
                "update_count": self.user_profile.get("update_count", 0),
                "has_profile": self.user_profile.get("profile_text", "None") != "None"
            },
            "user_knowledge": {
                "total_entries": len(self.user_knowledge),
                "capacity": self.knowledge_capacity,
                "usage_percentage": (len(self.user_knowledge) / self.knowledge_capacity) * 100
            },
            "assistant_knowledge": {
                "total_entries": len(self.assistant_knowledge),
                "capacity": self.knowledge_capacity,
                "usage_percentage": (len(self.assistant_knowledge) / self.knowledge_capacity) * 100
            }
        }
    
    def export_user_data(self) -> Dict[str, Any]:
        """Export all user data for backup or migration"""
        return {
            "user_id": self.user_id,
            "assistant_id": self.assistant_id,
            "export_timestamp": get_timestamp(),
            "user_profile": self.user_profile,
            "user_knowledge": self.user_knowledge,
            "assistant_knowledge": self.assistant_knowledge,
            "stats": self.get_memory_stats()
        }
    
    def clear_user_data(self, backup: bool = True) -> bool:
        """
        Clear all user data
        
        Args:
            backup: Whether to create a backup before clearing
            
        Returns:
            True if clearing was successful
        """
        try:
            if backup:
                backup_file = os.path.join(self.data_path, f"backup_{get_timestamp().replace(':', '-')}.json")
                backup_data = self.export_user_data()
                safe_json_save(backup_data, backup_file)
            
            # Reset to defaults
            self.user_profile = self._load_user_profile()  # This loads defaults
            self.user_knowledge = []
            self.assistant_knowledge = []
            self.user_embeddings = None
            self.assistant_embeddings = None
            
            # Save cleared state
            self._save_user_profile()
            self._save_user_knowledge()
            self._save_assistant_knowledge()
            
            # Remove embedding files
            for file_path in [self.user_embeddings_file, self.assistant_embeddings_file]:
                if os.path.exists(file_path):
                    os.remove(file_path)
            
            return True
        except Exception as e:
            print(f"Error clearing user data: {e}")
            return False
