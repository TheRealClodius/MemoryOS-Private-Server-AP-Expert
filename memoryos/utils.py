"""
Utility functions for MemoryOS
"""

import json
import os
import numpy as np
from datetime import datetime
from typing import Any, Dict, List, Optional
import hashlib


def get_timestamp() -> str:
    """Get current timestamp in ISO format"""
    return datetime.now().isoformat()


def ensure_directory_exists(path: str) -> None:
    """Ensure directory exists, create if not"""
    os.makedirs(path, exist_ok=True)


def safe_json_load(file_path: str, default: Any = None) -> Any:
    """Safely load JSON file with default fallback"""
    try:
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        print(f"Warning: Could not load {file_path}: {e}")
    return default if default is not None else {}


def safe_json_save(data: Any, file_path: str) -> bool:
    """Safely save data to JSON file"""
    try:
        ensure_directory_exists(os.path.dirname(file_path))
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return True
    except (IOError, TypeError) as e:
        print(f"Warning: Could not save {file_path}: {e}")
        return False


def compute_similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    """Compute cosine similarity between two embeddings"""
    try:
        if embedding1.size == 0 or embedding2.size == 0:
            return 0.0
        
        # Normalize embeddings
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        # Compute cosine similarity
        similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)
        return float(np.clip(similarity, -1.0, 1.0))
    except Exception as e:
        print(f"Error computing similarity: {e}")
        return 0.0


def generate_hash(text: str) -> str:
    """Generate SHA-256 hash of text"""
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def truncate_text(text: str, max_length: int = 1000) -> str:
    """Truncate text to maximum length"""
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."


def format_memory_entry(entry: Dict[str, Any]) -> str:
    """Format memory entry for display"""
    timestamp = entry.get('timestamp', 'Unknown time')
    content = entry.get('content', entry.get('user_input', ''))
    response = entry.get('agent_response', '')
    
    formatted = f"[{timestamp}]\n"
    if content:
        formatted += f"User: {content}\n"
    if response:
        formatted += f"Assistant: {response}\n"
    
    return formatted.strip()


def validate_embedding_dimension(embedding: np.ndarray, expected_dim: int = 1536) -> bool:
    """Validate embedding has expected dimensions"""
    return len(embedding.shape) == 1 and embedding.shape[0] == expected_dim


def normalize_embedding(embedding: np.ndarray) -> np.ndarray:
    """Normalize embedding to unit vector"""
    norm = np.linalg.norm(embedding)
    if norm == 0:
        return embedding
    return embedding / norm


def merge_dictionaries(dict1: Dict, dict2: Dict) -> Dict:
    """Safely merge two dictionaries"""
    result = dict1.copy()
    result.update(dict2)
    return result


def extract_keywords(text: str, max_keywords: int = 10) -> List[str]:
    """Extract simple keywords from text"""
    # Simple keyword extraction - split by common delimiters
    import re
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
    
    # Remove common stop words
    stop_words = {'the', 'and', 'are', 'for', 'with', 'this', 'that', 'was', 'will', 'have', 'has', 'had'}
    keywords = [word for word in words if word not in stop_words]
    
    # Return unique keywords up to max_keywords
    unique_keywords = list(dict.fromkeys(keywords))
    return unique_keywords[:max_keywords]


def calculate_text_stats(text: str) -> Dict[str, int]:
    """Calculate basic text statistics"""
    return {
        'length': len(text),
        'words': len(text.split()),
        'lines': text.count('\n') + 1,
        'characters': len(text.replace(' ', ''))
    }


def safe_float_conversion(value: Any, default: float = 0.0) -> float:
    """Safely convert value to float"""
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def batch_process(items: List[Any], batch_size: int = 100) -> List[List[Any]]:
    """Split items into batches"""
    batches = []
    for i in range(0, len(items), batch_size):
        batches.append(items[i:i + batch_size])
    return batches


def is_valid_timestamp(timestamp_str: str) -> bool:
    """Validate timestamp string format"""
    try:
        datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
        return True
    except ValueError:
        return False


def clean_text(text: str) -> str:
    """Clean text by removing extra whitespace and normalizing"""
    import re
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove leading/trailing whitespace
    text = text.strip()
    return text
