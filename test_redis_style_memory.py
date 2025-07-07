#!/usr/bin/env python3
"""
Test Redis-style short-term memory implementation
"""

import sys
import os
from pathlib import Path

# Add the current directory to sys.path to import memoryos
sys.path.insert(0, str(Path(__file__).parent))

from memoryos.short_term import ShortTermMemory
from memoryos.utils import get_timestamp

def test_redis_style_performance():
    """Test Redis-style performance characteristics"""
    print("🚀 Testing Redis-style Short-Term Memory Performance")
    print("=" * 60)
    
    # Initialize with small capacity for testing
    short_term = ShortTermMemory(
        user_id="test_user",
        data_path="./test_memoryos_data",
        capacity=3  # Small capacity to test overflow
    )
    
    print(f"📊 Initial state: {len(short_term.memory)} entries, capacity: {short_term.capacity}")
    
    # Test 1: Add entries and test automatic eviction
    print("\n1. Testing automatic FIFO eviction...")
    test_entries = [
        ("What is Python?", "Python is a programming language."),
        ("How do I learn coding?", "Start with the basics and practice regularly."),
        ("What is AI?", "AI stands for Artificial Intelligence."),
        ("What is machine learning?", "ML is a subset of AI that learns from data."),  # This should evict first entry
        ("What is deep learning?", "Deep learning uses neural networks."),  # This should evict second entry
    ]
    
    for i, (user_input, agent_response) in enumerate(test_entries):
        print(f"   Adding entry {i+1}: {user_input[:30]}...")
        entry = short_term.add_qa_pair(user_input, agent_response)
        print(f"   Memory size: {len(short_term.memory)}/{short_term.capacity}")
        
        # Check if overflow was handled
        if len(short_term.memory) == short_term.capacity and i >= short_term.capacity:
            overflow_entries = short_term.get_overflow_entries()
            print(f"   Overflow entries: {len(overflow_entries)}")
    
    # Test 2: Verify FIFO behavior
    print("\n2. Verifying FIFO behavior...")
    current_entries = short_term.get_all()
    print(f"   Current entries in memory: {len(current_entries)}")
    for i, entry in enumerate(current_entries):
        print(f"   Entry {i+1}: {entry['user_input'][:40]}...")
    
    # Test 3: Test Redis-style operations
    print("\n3. Testing Redis-style operations...")
    
    # Test is_full
    print(f"   Is memory full? {short_term.is_full()}")
    
    # Test pop_oldest
    if not short_term.is_empty():
        oldest = short_term.pop_oldest()
        print(f"   Popped oldest: {oldest['user_input'][:40]}...")
        print(f"   Memory size after pop: {len(short_term.memory)}")
    
    # Test 4: Performance characteristics
    print("\n4. Testing performance characteristics...")
    
    # Test fast retrieval
    recent_entries = short_term.get_recent(2)
    print(f"   Retrieved {len(recent_entries)} recent entries")
    
    # Test keyword search (should be fast with deque)
    matches = short_term.search_by_keyword("learning")
    print(f"   Found {len(matches)} matches for 'learning'")
    
    # Test 5: Verify file persistence
    print("\n5. Testing file persistence...")
    
    # Create new instance to test loading
    short_term2 = ShortTermMemory(
        user_id="test_user",
        data_path="./test_memoryos_data",
        capacity=3
    )
    print(f"   Loaded {len(short_term2.memory)} entries from file")
    
    # Test 6: Memory statistics
    print("\n6. Memory statistics...")
    stats = short_term.get_memory_stats()
    print(f"   Total entries: {stats['total_entries']}")
    print(f"   Usage: {stats['usage_percentage']:.1f}%")
    print(f"   Is full: {short_term.is_full()}")
    print(f"   Is empty: {short_term.is_empty()}")
    
    print("\n✅ Redis-style Short-Term Memory Test Complete!")
    return True

def main():
    """Run Redis-style memory tests"""
    try:
        result = test_redis_style_performance()
        if result:
            print("\n🎉 All Redis-style tests passed!")
        else:
            print("\n❌ Some tests failed!")
    except Exception as e:
        print(f"\n💥 Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()