"""
Mid-term memory management for MemoryOS
"""

import json
import os
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from .utils import (
    get_timestamp, safe_json_save, safe_json_load, ensure_directory_exists,
    compute_similarity, generate_hash, format_memory_entry
)


class MidTermMemory:
    """Manages mid-term memory storage with heat-based consolidation"""
    
    def __init__(self, user_id: str, data_path: str, capacity: int = 2000, heat_threshold: float = 5.0):
        """
        Initialize mid-term memory
        
        Args:
            user_id: Unique identifier for the user
            data_path: Base path for data storage
            capacity: Maximum number of memory segments
            heat_threshold: Heat threshold for promoting to long-term memory
        """
        self.user_id = user_id
        self.capacity = capacity
        self.heat_threshold = heat_threshold
        self.data_path = os.path.join(data_path, user_id, "mid_term")
        ensure_directory_exists(self.data_path)
        
        self.memory_file = os.path.join(self.data_path, "memory.json")
        self.embeddings_file = os.path.join(self.data_path, "embeddings.npy")
        
        self.memory_segments: List[Dict[str, Any]] = self._load_memory()
        self.embeddings: Optional[np.ndarray] = self._load_embeddings()
    
    def _load_memory(self) -> List[Dict[str, Any]]:
        """Load memory segments from file"""
        return safe_json_load(self.memory_file, [])
    
    def _save_memory(self) -> bool:
        """Save memory segments to file"""
        return safe_json_save(self.memory_segments, self.memory_file)
    
    def _load_embeddings(self) -> Optional[np.ndarray]:
        """Load embeddings from file"""
        try:
            if os.path.exists(self.embeddings_file):
                return np.load(self.embeddings_file)
        except Exception as e:
            print(f"Error loading embeddings: {e}")
        return None
    
    def _save_embeddings(self) -> bool:
        """Save embeddings to file"""
        try:
            if self.embeddings is not None:
                np.save(self.embeddings_file, self.embeddings)
                return True
        except Exception as e:
            print(f"Error saving embeddings: {e}")
        return False
    
    def add_consolidated_segment(
        self,
        qa_pairs: List[Dict[str, Any]],
        embedding: np.ndarray,
        summary: str,
        themes: List[str] = None,
        meta_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Add a consolidated memory segment from short-term memory
        
        Args:
            qa_pairs: List of QA pairs that form this segment
            embedding: Embedding representation of the segment
            summary: Summary of the segment content
            themes: Main themes/topics in the segment
            meta_data: Additional metadata
            
        Returns:
            The created memory segment
        """
        timestamp = get_timestamp()
        segment_id = generate_hash(f"{self.user_id}_{timestamp}_{len(self.memory_segments)}")
        
        segment = {
            "id": segment_id,
            "user_id": self.user_id,
            "timestamp": timestamp,
            "qa_pairs": qa_pairs,
            "summary": summary,
            "themes": themes or [],
            "heat": 1.0,  # Initial heat
            "access_count": 0,
            "last_accessed": timestamp,
            "meta_data": meta_data or {},
            "quality_score": self._calculate_quality_score(qa_pairs, summary),
            "importance_score": self._calculate_importance_score(qa_pairs)
        }
        
        # Add to memory
        self.memory_segments.append(segment)
        
        # Add embedding
        if self.embeddings is None:
            self.embeddings = embedding.reshape(1, -1)
        else:
            self.embeddings = np.vstack([self.embeddings, embedding.reshape(1, -1)])
        
        # Maintain capacity
        if len(self.memory_segments) > self.capacity:
            self._handle_overflow()
        
        self._save_memory()
        self._save_embeddings()
        
        return segment
    
    def _calculate_quality_score(self, qa_pairs: List[Dict[str, Any]], summary: str) -> float:
        """Calculate quality score for a memory segment"""
        score = 0.0
        
        # Length and content quality
        total_length = sum(len(qa.get("user_input", "") + qa.get("agent_response", "")) for qa in qa_pairs)
        if total_length > 100:
            score += 1.0
        
        # Number of QA pairs
        score += min(len(qa_pairs) * 0.2, 1.0)
        
        # Summary quality
        if len(summary) > 50:
            score += 0.5
        
        # Interaction engagement (based on response length)
        avg_response_length = np.mean([len(qa.get("agent_response", "")) for qa in qa_pairs])
        if avg_response_length > 50:
            score += 0.5
        
        return min(score, 5.0)  # Cap at 5.0
    
    def _calculate_importance_score(self, qa_pairs: List[Dict[str, Any]]) -> float:
        """Calculate importance score based on content analysis"""
        importance_keywords = [
            'important', 'remember', 'please', 'help', 'need', 'problem',
            'question', 'urgent', 'critical', 'essential', 'key'
        ]
        
        score = 0.0
        total_text = ""
        
        for qa in qa_pairs:
            total_text += qa.get("user_input", "") + " " + qa.get("agent_response", "")
        
        total_text = total_text.lower()
        
        # Count importance keywords
        keyword_count = sum(total_text.count(keyword) for keyword in importance_keywords)
        score += min(keyword_count * 0.2, 2.0)
        
        # Question indicators
        question_count = total_text.count('?')
        score += min(question_count * 0.1, 1.0)
        
        # Personal information indicators
        personal_keywords = ['my', 'i', 'me', 'myself', 'personal', 'private']
        personal_count = sum(total_text.count(keyword) for keyword in personal_keywords)
        score += min(personal_count * 0.05, 1.0)
        
        return min(score, 5.0)  # Cap at 5.0
    
    def _handle_overflow(self) -> None:
        """Handle memory overflow by removing least important segments"""
        if len(self.memory_segments) <= self.capacity:
            return
        
        # Sort by combined score (heat + quality + importance)
        segments_with_scores = []
        for i, segment in enumerate(self.memory_segments):
            combined_score = (
                segment.get("heat", 0.0) +
                segment.get("quality_score", 0.0) +
                segment.get("importance_score", 0.0)
            )
            segments_with_scores.append((combined_score, i, segment))
        
        segments_with_scores.sort(key=lambda x: x[0])  # Sort by score ascending
        
        # Remove lowest scoring segments
        to_remove = len(self.memory_segments) - self.capacity
        removed_indices = []
        
        for i in range(to_remove):
            _, idx, segment = segments_with_scores[i]
            removed_indices.append(idx)
            
            # Archive removed segment
            self._archive_segment(segment)
        
        # Remove from memory and embeddings
        removed_indices.sort(reverse=True)  # Remove from end to avoid index shifts
        for idx in removed_indices:
            self.memory_segments.pop(idx)
            if self.embeddings is not None:
                self.embeddings = np.delete(self.embeddings, idx, axis=0)
    
    def _archive_segment(self, segment: Dict[str, Any]) -> None:
        """Archive a removed segment"""
        archive_file = os.path.join(self.data_path, "archived_segments.json")
        archived = safe_json_load(archive_file, [])
        
        segment["archived_at"] = get_timestamp()
        archived.append(segment)
        
        safe_json_save(archived, archive_file)
    
    def update_heat(self, segment_id: str, heat_increase: float = 1.0) -> bool:
        """
        Update heat for a memory segment
        
        Args:
            segment_id: ID of the segment to update
            heat_increase: Amount to increase heat by
            
        Returns:
            True if update was successful
        """
        for segment in self.memory_segments:
            if segment.get("id") == segment_id:
                segment["heat"] = segment.get("heat", 0.0) + heat_increase
                segment["access_count"] = segment.get("access_count", 0) + 1
                segment["last_accessed"] = get_timestamp()
                self._save_memory()
                return True
        return False
    
    def get_hot_segments(self, threshold: Optional[float] = None) -> List[Dict[str, Any]]:
        """
        Get segments that exceed the heat threshold
        
        Args:
            threshold: Heat threshold (uses instance threshold if not provided)
            
        Returns:
            List of hot segments ready for long-term promotion
        """
        threshold = threshold or self.heat_threshold
        hot_segments = []
        
        for segment in self.memory_segments:
            if segment.get("heat", 0.0) >= threshold:
                hot_segments.append(segment)
        
        return hot_segments
    
    def search_by_embedding(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        similarity_threshold: float = 0.3
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Search memory segments by embedding similarity
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of top results to return
            similarity_threshold: Minimum similarity threshold
            
        Returns:
            List of (segment, similarity_score) tuples
        """
        if self.embeddings is None or len(self.memory_segments) == 0:
            return []
        
        similarities = []
        for i, segment_embedding in enumerate(self.embeddings):
            similarity = compute_similarity(query_embedding, segment_embedding)
            if similarity >= similarity_threshold:
                similarities.append((i, similarity))
        
        # Sort by similarity and get top k
        similarities.sort(key=lambda x: x[1], reverse=True)
        results = []
        
        for i, similarity in similarities[:top_k]:
            segment = self.memory_segments[i]
            # Update access information
            segment["access_count"] = segment.get("access_count", 0) + 1
            segment["last_accessed"] = get_timestamp()
            # Increase heat slightly for access
            segment["heat"] = segment.get("heat", 0.0) + 0.1
            
            results.append((segment, similarity))
        
        if results:
            self._save_memory()
        
        return results
    
    def search_by_themes(self, themes: List[str], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search memory segments by themes
        
        Args:
            themes: List of themes to search for
            top_k: Number of top results to return
            
        Returns:
            List of matching segments
        """
        theme_set = set(theme.lower() for theme in themes)
        scored_segments = []
        
        for segment in self.memory_segments:
            segment_themes = set(theme.lower() for theme in segment.get("themes", []))
            overlap = len(theme_set.intersection(segment_themes))
            
            if overlap > 0:
                score = overlap / len(theme_set)  # Normalized overlap score
                scored_segments.append((score, segment))
        
        # Sort by score and return top k
        scored_segments.sort(key=lambda x: x[0], reverse=True)
        return [segment for _, segment in scored_segments[:top_k]]
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get statistics about mid-term memory"""
        if not self.memory_segments:
            return {
                "total_segments": 0,
                "capacity": self.capacity,
                "usage_percentage": 0.0,
                "average_heat": 0.0,
                "hot_segments_count": 0,
                "themes_distribution": {}
            }
        
        heats = [segment.get("heat", 0.0) for segment in self.memory_segments]
        hot_count = len([h for h in heats if h >= self.heat_threshold])
        
        # Count themes
        themes_count = {}
        for segment in self.memory_segments:
            for theme in segment.get("themes", []):
                themes_count[theme] = themes_count.get(theme, 0) + 1
        
        return {
            "total_segments": len(self.memory_segments),
            "capacity": self.capacity,
            "usage_percentage": (len(self.memory_segments) / self.capacity) * 100,
            "average_heat": np.mean(heats),
            "max_heat": np.max(heats),
            "hot_segments_count": hot_count,
            "themes_distribution": themes_count
        }
    
    def promote_segment(self, segment_id: str) -> Optional[Dict[str, Any]]:
        """
        Remove and return a segment for promotion to long-term memory
        
        Args:
            segment_id: ID of the segment to promote
            
        Returns:
            The promoted segment or None if not found
        """
        for i, segment in enumerate(self.memory_segments):
            if segment.get("id") == segment_id:
                # Remove from mid-term memory
                promoted_segment = self.memory_segments.pop(i)
                
                # Remove corresponding embedding
                if self.embeddings is not None and i < len(self.embeddings):
                    self.embeddings = np.delete(self.embeddings, i, axis=0)
                
                self._save_memory()
                self._save_embeddings()
                
                return promoted_segment
        
        return None
    
    def get_recent_segments(self, hours: int = 168) -> List[Dict[str, Any]]:  # Default 1 week
        """Get segments created within specified hours"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_segments = []
        
        for segment in self.memory_segments:
            try:
                segment_time = datetime.fromisoformat(segment["timestamp"].replace('Z', '+00:00'))
                if segment_time >= cutoff_time:
                    recent_segments.append(segment)
            except (ValueError, KeyError):
                continue
        
        return recent_segments
    
    def consolidate_similar_segments(self, similarity_threshold: float = 0.8) -> int:
        """
        Consolidate segments that are very similar
        
        Args:
            similarity_threshold: Threshold for considering segments similar
            
        Returns:
            Number of segments consolidated
        """
        if self.embeddings is None or len(self.memory_segments) < 2:
            return 0
        
        consolidated_count = 0
        to_remove = []
        
        for i in range(len(self.memory_segments)):
            if i in to_remove:
                continue
                
            for j in range(i + 1, len(self.memory_segments)):
                if j in to_remove:
                    continue
                
                similarity = compute_similarity(self.embeddings[i], self.embeddings[j])
                
                if similarity >= similarity_threshold:
                    # Merge segments
                    segment_i = self.memory_segments[i]
                    segment_j = self.memory_segments[j]
                    
                    # Combine QA pairs
                    segment_i["qa_pairs"].extend(segment_j["qa_pairs"])
                    
                    # Combine themes
                    all_themes = set(segment_i.get("themes", []) + segment_j.get("themes", []))
                    segment_i["themes"] = list(all_themes)
                    
                    # Update summary
                    segment_i["summary"] += f" | {segment_j['summary']}"
                    
                    # Update heat and scores
                    segment_i["heat"] = max(segment_i.get("heat", 0), segment_j.get("heat", 0))
                    segment_i["access_count"] = segment_i.get("access_count", 0) + segment_j.get("access_count", 0)
                    
                    # Mark for removal
                    to_remove.append(j)
                    consolidated_count += 1
        
        # Remove consolidated segments
        to_remove.sort(reverse=True)
        for idx in to_remove:
            self.memory_segments.pop(idx)
            if self.embeddings is not None:
                self.embeddings = np.delete(self.embeddings, idx, axis=0)
        
        if consolidated_count > 0:
            self._save_memory()
            self._save_embeddings()
        
        return consolidated_count
