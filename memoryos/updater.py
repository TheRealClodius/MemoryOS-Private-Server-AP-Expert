"""
Memory updater system for MemoryOS
"""

import numpy as np
from typing import Dict, List, Optional, Any
from .utils import get_timestamp
from .prompts import (
    MEMORY_CONSOLIDATION_PROMPT, USER_PROFILE_ANALYSIS_PROMPT,
    KNOWLEDGE_EXTRACTION_PROMPT, ASSISTANT_KNOWLEDGE_PROMPT
)


class MemoryUpdater:
    """Handles memory updates and consolidation between layers"""
    
    def __init__(self, short_term_memory, mid_term_memory, long_term_memory, 
                 embedding_function, llm_function):
        """
        Initialize memory updater
        
        Args:
            short_term_memory: ShortTermMemory instance
            mid_term_memory: MidTermMemory instance  
            long_term_memory: LongTermMemory instance
            embedding_function: Function to generate embeddings
            llm_function: Function to call LLM for text processing
        """
        self.short_term_memory = short_term_memory
        self.mid_term_memory = mid_term_memory
        self.long_term_memory = long_term_memory
        self.embedding_function = embedding_function
        self.llm_function = llm_function
    
    def process_short_term_overflow(self) -> bool:
        """
        Process overflow from short-term memory and consolidate into mid-term
        
        Returns:
            True if processing was successful
        """
        try:
            # Get overflow entries
            overflow_entries = self.short_term_memory.get_overflow_entries()
            
            if not overflow_entries:
                return True
            
            # Consolidate entries into segments
            segments = self._consolidate_qa_pairs(overflow_entries)
            
            # Process each segment
            for segment_qa_pairs in segments:
                self._create_mid_term_segment(segment_qa_pairs)
            
            # Clear overflow after successful processing
            self.short_term_memory.clear_overflow()
            
            return True
        
        except Exception as e:
            print(f"Error processing short-term overflow: {e}")
            return False
    
    def _consolidate_qa_pairs(self, qa_pairs: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """
        Group QA pairs into coherent segments for consolidation
        
        Args:
            qa_pairs: List of QA pairs to consolidate
            
        Returns:
            List of QA pair groups (segments)
        """
        # Simple grouping by time proximity and content similarity
        # In a more sophisticated implementation, you could use embedding similarity
        
        segments = []
        current_segment = []
        
        for qa_pair in qa_pairs:
            if not current_segment:
                current_segment.append(qa_pair)
            else:
                # Simple heuristic: group if within reasonable length
                if len(current_segment) < 5:  # Max 5 QA pairs per segment
                    current_segment.append(qa_pair)
                else:
                    segments.append(current_segment)
                    current_segment = [qa_pair]
        
        if current_segment:
            segments.append(current_segment)
        
        return segments
    
    def _create_mid_term_segment(self, qa_pairs: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        Create a mid-term memory segment from QA pairs
        
        Args:
            qa_pairs: List of QA pairs to form the segment
            
        Returns:
            Created segment or None if failed
        """
        try:
            # Generate summary using LLM
            conversation_text = self._format_qa_pairs_for_llm(qa_pairs)
            summary_prompt = MEMORY_CONSOLIDATION_PROMPT.format(
                conversation_segments=conversation_text
            )
            
            summary = self.llm_function(summary_prompt)
            
            # Extract themes (simple keyword extraction for now)
            themes = self._extract_themes(qa_pairs, summary)
            
            # Generate embedding for the segment
            segment_text = f"{summary} {' '.join(themes)}"
            embedding = self.embedding_function(segment_text)
            
            # Create segment in mid-term memory
            segment = self.mid_term_memory.add_consolidated_segment(
                qa_pairs=qa_pairs,
                embedding=embedding,
                summary=summary,
                themes=themes,
                meta_data={
                    "consolidation_timestamp": get_timestamp(),
                    "source": "short_term_overflow"
                }
            )
            
            return segment
        
        except Exception as e:
            print(f"Error creating mid-term segment: {e}")
            return None
    
    def _format_qa_pairs_for_llm(self, qa_pairs: List[Dict[str, Any]]) -> str:
        """Format QA pairs for LLM processing"""
        formatted_text = ""
        for i, qa in enumerate(qa_pairs, 1):
            user_input = qa.get("user_input", "")
            agent_response = qa.get("agent_response", "")
            timestamp = qa.get("timestamp", "")
            
            formatted_text += f"\n--- Interaction {i} ({timestamp}) ---\n"
            formatted_text += f"User: {user_input}\n"
            formatted_text += f"Assistant: {agent_response}\n"
        
        return formatted_text
    
    def _extract_themes(self, qa_pairs: List[Dict[str, Any]], summary: str) -> List[str]:
        """Extract themes from QA pairs and summary"""
        # Simple theme extraction based on common words
        all_text = summary.lower()
        for qa in qa_pairs:
            all_text += " " + qa.get("user_input", "").lower()
            all_text += " " + qa.get("agent_response", "").lower()
        
        # Extract meaningful words (simple approach)
        words = all_text.split()
        word_counts = {}
        
        # Filter out common stop words
        stop_words = {
            'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
            'by', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these',
            'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him',
            'her', 'us', 'them', 'my', 'your', 'his', 'our', 'their'
        }
        
        for word in words:
            word = word.strip('.,!?;:"()[]{}')
            if len(word) > 3 and word not in stop_words:
                word_counts[word] = word_counts.get(word, 0) + 1
        
        # Get top themes
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        themes = [word for word, count in sorted_words[:5] if count > 1]
        
        return themes
    
    def process_hot_segments(self) -> bool:
        """
        Process hot segments from mid-term memory for long-term promotion
        
        Returns:
            True if processing was successful
        """
        try:
            hot_segments = self.mid_term_memory.get_hot_segments()
            
            for segment in hot_segments:
                self._process_hot_segment(segment)
            
            return True
        
        except Exception as e:
            print(f"Error processing hot segments: {e}")
            return False
    
    def _process_hot_segment(self, segment: Dict[str, Any]) -> bool:
        """
        Process a single hot segment for long-term memory extraction
        
        Args:
            segment: The hot segment to process
            
        Returns:
            True if processing was successful
        """
        try:
            # Extract user profile information
            self._extract_user_profile_info(segment)
            
            # Extract user knowledge
            self._extract_user_knowledge(segment)
            
            # Extract assistant knowledge
            self._extract_assistant_knowledge(segment)
            
            # Promote segment (remove from mid-term after processing)
            self.mid_term_memory.promote_segment(segment.get("id"))
            
            return True
        
        except Exception as e:
            print(f"Error processing hot segment {segment.get('id', 'unknown')}: {e}")
            return False
    
    def _extract_user_profile_info(self, segment: Dict[str, Any]) -> bool:
        """Extract and update user profile information from segment"""
        try:
            conversation_text = self._format_qa_pairs_for_llm(segment.get("qa_pairs", []))
            
            profile_prompt = USER_PROFILE_ANALYSIS_PROMPT.format(
                conversation_history=conversation_text
            )
            
            new_profile_analysis = self.llm_function(profile_prompt)
            
            # Get current profile
            current_profile = self.long_term_memory.get_user_profile_summary()
            
            # If current profile is empty or "None", use new analysis directly
            if current_profile == "None" or not current_profile.strip():
                updated_profile = new_profile_analysis
            else:
                # Merge with existing profile
                merge_prompt = f"""
                Current Profile: {current_profile}
                
                New Information: {new_profile_analysis}
                
                Merge these profiles into a comprehensive, updated user profile that incorporates all relevant information:
                """
                updated_profile = self.llm_function(merge_prompt)
            
            # Update user profile
            return self.long_term_memory.update_user_profile(updated_profile)
        
        except Exception as e:
            print(f"Error extracting user profile info: {e}")
            return False
    
    def _extract_user_knowledge(self, segment: Dict[str, Any]) -> bool:
        """Extract user knowledge from segment"""
        try:
            qa_pairs = segment.get("qa_pairs", [])
            
            for qa_pair in qa_pairs:
                user_input = qa_pair.get("user_input", "")
                agent_response = qa_pair.get("agent_response", "")
                
                knowledge_prompt = KNOWLEDGE_EXTRACTION_PROMPT.format(
                    user_input=user_input,
                    agent_response=agent_response
                )
                
                extracted_knowledge = self.llm_function(knowledge_prompt)
                
                # Split into individual knowledge points
                knowledge_points = self._parse_knowledge_points(extracted_knowledge)
                
                for knowledge in knowledge_points:
                    if knowledge.strip():
                        embedding = self.embedding_function(knowledge)
                        self.long_term_memory.add_user_knowledge(
                            knowledge=knowledge,
                            embedding=embedding,
                            source="mid_term_promotion",
                            confidence=0.8
                        )
            
            return True
        
        except Exception as e:
            print(f"Error extracting user knowledge: {e}")
            return False
    
    def _extract_assistant_knowledge(self, segment: Dict[str, Any]) -> bool:
        """Extract assistant knowledge from segment"""
        try:
            qa_pairs = segment.get("qa_pairs", [])
            
            for qa_pair in qa_pairs:
                user_input = qa_pair.get("user_input", "")
                agent_response = qa_pair.get("agent_response", "")
                
                assistant_prompt = ASSISTANT_KNOWLEDGE_PROMPT.format(
                    user_input=user_input,
                    agent_response=agent_response
                )
                
                extracted_knowledge = self.llm_function(assistant_prompt)
                
                # Split into individual knowledge points
                knowledge_points = self._parse_knowledge_points(extracted_knowledge)
                
                for knowledge in knowledge_points:
                    if knowledge.strip():
                        embedding = self.embedding_function(knowledge)
                        self.long_term_memory.add_assistant_knowledge(
                            knowledge=knowledge,
                            embedding=embedding,
                            source="mid_term_promotion",
                            confidence=0.8
                        )
            
            return True
        
        except Exception as e:
            print(f"Error extracting assistant knowledge: {e}")
            return False
    
    def _parse_knowledge_points(self, knowledge_text: str) -> List[str]:
        """Parse knowledge text into individual points"""
        # Simple parsing - split by newlines and filter out empty/short lines
        lines = knowledge_text.split('\n')
        knowledge_points = []
        
        for line in lines:
            line = line.strip()
            # Remove numbering, bullets, etc.
            line = line.lstrip('0123456789.-• ')
            
            if len(line) > 10:  # Only keep substantial knowledge points
                knowledge_points.append(line)
        
        return knowledge_points
    
    def update_memory_heat(self, access_patterns: Dict[str, Any]) -> bool:
        """
        Update memory heat based on access patterns
        
        Args:
            access_patterns: Dictionary containing access information
            
        Returns:
            True if update was successful
        """
        try:
            # Update heat for mid-term memory segments based on access
            for segment_id, access_info in access_patterns.items():
                heat_increase = self._calculate_heat_increase(access_info)
                self.mid_term_memory.update_heat(segment_id, heat_increase)
            
            return True
        
        except Exception as e:
            print(f"Error updating memory heat: {e}")
            return False
    
    def _calculate_heat_increase(self, access_info: Dict[str, Any]) -> float:
        """Calculate heat increase based on access information"""
        base_heat = 0.5
        
        # Increase heat based on recency
        access_count = access_info.get("access_count", 0)
        recency_factor = access_info.get("recency_factor", 1.0)  # Higher for recent access
        
        heat_increase = base_heat + (access_count * 0.1) + (recency_factor * 0.3)
        
        return min(heat_increase, 2.0)  # Cap heat increase
    
    def consolidate_similar_memories(self) -> Dict[str, int]:
        """
        Consolidate similar memories across layers
        
        Returns:
            Dictionary with consolidation statistics
        """
        stats = {
            "mid_term_consolidated": 0,
            "user_knowledge_deduplicated": 0,
            "assistant_knowledge_deduplicated": 0
        }
        
        try:
            # Consolidate similar mid-term segments
            stats["mid_term_consolidated"] = self.mid_term_memory.consolidate_similar_segments()
            
            # For knowledge deduplication, this would require more sophisticated
            # similarity checking that's not implemented in the base classes
            # but could be added as an enhancement
            
        except Exception as e:
            print(f"Error consolidating memories: {e}")
        
        return stats
    
    def get_update_statistics(self) -> Dict[str, Any]:
        """Get statistics about memory updates"""
        try:
            short_term_stats = self.short_term_memory.get_memory_stats()
            mid_term_stats = self.mid_term_memory.get_memory_stats()
            long_term_stats = self.long_term_memory.get_memory_stats()
            
            return {
                "timestamp": get_timestamp(),
                "short_term_usage": short_term_stats.get("usage_percentage", 0),
                "mid_term_usage": mid_term_stats.get("usage_percentage", 0),
                "hot_segments_count": mid_term_stats.get("hot_segments_count", 0),
                "user_knowledge_count": long_term_stats.get("user_knowledge", {}).get("total_entries", 0),
                "assistant_knowledge_count": long_term_stats.get("assistant_knowledge", {}).get("total_entries", 0),
                "consolidation_needed": short_term_stats.get("usage_percentage", 0) >= 80
            }
        
        except Exception as e:
            print(f"Error getting update statistics: {e}")
            return {"error": str(e), "timestamp": get_timestamp()}
