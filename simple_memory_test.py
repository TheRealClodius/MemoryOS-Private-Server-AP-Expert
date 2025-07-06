#!/usr/bin/env python3
"""
Simple test to verify memory storage and retrieval without API calls
"""

import sys
import os
from pathlib import Path

# Add the current directory to sys.path to import memoryos
sys.path.insert(0, str(Path(__file__).parent))

from memoryos.short_term import ShortTermMemory
from memoryos.utils import get_timestamp

def test_short_term_memory():
    """Test short-term memory storage and retrieval"""
    print("🧪 Testing Short-Term Memory Storage and Retrieval")
    print("=" * 50)
    
    # Initialize short-term memory
    short_term = ShortTermMemory(
        user_id="demo_user",
        data_path="./memoryos_data",
        capacity=10
    )
    
    # Test 1: Add new memory
    print("\n1. Adding new memory...")
    result = short_term.add_qa_pair(
        user_input="What is the weather like today?",
        agent_response="I don't have access to real-time weather data, but I can help you find weather information from reliable sources."
    )
    print(f"   Added memory: {result['user_input'][:50]}...")
    
    # Test 2: Get all memories
    print("\n2. Retrieving all memories...")
    all_memories = short_term.get_all()
    print(f"   Found {len(all_memories)} memories")
    
    for i, memory in enumerate(all_memories):
        print(f"   Memory {i+1}: {memory['user_input'][:50]}...")
    
    # Test 3: Get memory stats
    print("\n3. Getting memory statistics...")
    stats = short_term.get_memory_stats()
    print(f"   Total entries: {stats['total_entries']}")
    print(f"   Usage: {stats['usage_percentage']:.1f}%")
    
    # Test 4: Test retrieval by query context
    print("\n4. Testing query-based retrieval...")
    context = short_term.get_context_for_query("weather", max_entries=3)
    print(f"   Found {len(context)} relevant memories for 'weather'")
    
    return len(all_memories) > 0

def test_memory_retrieval_specific_user():
    """Test retrieval for specific user/assistant combination"""
    print("\n🧪 Testing User-Specific Memory Retrieval")
    print("=" * 50)
    
    # Test demo_user memories
    demo_short_term = ShortTermMemory(
        user_id="demo_user",
        data_path="./memoryos_data",
        capacity=10
    )
    
    demo_memories = demo_short_term.get_all()
    print(f"Demo user memories: {len(demo_memories)}")
    
    for i, memory in enumerate(demo_memories):
        print(f"   Memory {i+1}: {memory['user_input'][:50]}...")
        print(f"   Response: {memory['agent_response'][:50]}...")
        print(f"   Timestamp: {memory['timestamp']}")
        print()
    
    return len(demo_memories) > 0

def main():
    print("🚀 Simple Memory Test Suite")
    print("=" * 50)
    
    results = []
    
    # Test 1: Short-term memory
    try:
        results.append(("Short-term Memory", test_short_term_memory()))
    except Exception as e:
        print(f"❌ Short-term memory test failed: {e}")
        results.append(("Short-term Memory", False))
    
    # Test 2: User-specific retrieval
    try:
        results.append(("User-specific Retrieval", test_memory_retrieval_specific_user()))
    except Exception as e:
        print(f"❌ User-specific retrieval test failed: {e}")
        results.append(("User-specific Retrieval", False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    passed = 0
    for test_name, passed_test in results:
        status = "✅ PASS" if passed_test else "❌ FAIL"
        print(f"   {test_name}: {status}")
        if passed_test:
            passed += 1
    
    print(f"\n🏆 Overall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 All tests passed! Memory system is working correctly.")
        return True
    else:
        print("⚠️ Some tests failed. Memory system needs investigation.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 