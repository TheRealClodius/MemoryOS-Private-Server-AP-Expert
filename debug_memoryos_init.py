#!/usr/bin/env python3
"""
Debug MemoryOS initialization to identify the exact parameter issue
"""

import os
from memoryos import Memoryos

def test_memoryos_init():
    """Test MemoryOS initialization with different parameter combinations"""
    
    # Test 1: Basic parameters
    print("Test 1: Basic MemoryOS initialization")
    try:
        memory = Memoryos(
            user_id="debug_test",
            openai_api_key=os.getenv("OPENAI_API_KEY", "test-key"),
            data_storage_path="./debug_test_data"
        )
        print("✅ Basic initialization successful")
        return True
    except Exception as e:
        print(f"❌ Basic initialization failed: {e}")
        return False

if __name__ == "__main__":
    success = test_memoryos_init()
    if success:
        print("MemoryOS initialization working correctly")
    else:
        print("MemoryOS initialization has issues")