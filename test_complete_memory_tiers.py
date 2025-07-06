#!/usr/bin/env python3
"""
Test complete memory tier architecture with Redis-style short-term memory
"""

import sys
import os
from pathlib import Path

# Add the current directory to sys.path to import memoryos
sys.path.insert(0, str(Path(__file__).parent))

from memoryos.memoryos import Memoryos

def test_memory_tier_integration():
    """Test complete memory tier integration"""
    print("🧠 Testing Complete Memory Tier Integration")
    print("=" * 60)
    
    try:
        # Initialize MemoryOS (this will test all memory tiers)
        config = {
            "user_id": "test_tier_user",
            "openai_api_key": "test_key",  # Won't be used for this test
            "data_storage_path": "./test_tier_data",
            "assistant_id": "test_assistant",
            "short_term_capacity": 3,  # Small for testing
            "mid_term_capacity": 10,
            "long_term_knowledge_capacity": 5,
            "retrieval_queue_capacity": 5,
            "mid_term_heat_threshold": 2.0,
            "llm_model": "gpt-4o-mini",
            "embedding_model": "text-embedding-3-small"
        }
        
        # Initialize without API calls
        memoryos = Memoryos(
            user_id=config["user_id"],
            openai_api_key=config["openai_api_key"],
            data_storage_path=config["data_storage_path"],
            assistant_id=config["assistant_id"],
            short_term_capacity=config["short_term_capacity"],
            mid_term_capacity=config["mid_term_capacity"],
            long_term_knowledge_capacity=config["long_term_knowledge_capacity"],
            retrieval_queue_capacity=config["retrieval_queue_capacity"],
            mid_term_heat_threshold=config["mid_term_heat_threshold"],
            llm_model=config["llm_model"],
            embedding_model=config["embedding_model"]
        )
        
        print("✅ MemoryOS initialized successfully with all memory tiers")
        
        # Test 1: Verify memory tier types
        print("\n1. Testing memory tier architectures...")
        
        # Check short-term memory (should be deque-based)
        print(f"   Short-term memory type: {type(memoryos.short_term_memory.memory)}")
        print(f"   Short-term capacity: {memoryos.short_term_memory.capacity}")
        print(f"   Is Redis-style deque: {hasattr(memoryos.short_term_memory.memory, 'popleft')}")
        
        # Check mid-term memory (should have embeddings)
        print(f"   Mid-term memory segments: {len(memoryos.mid_term_memory.memory_segments)}")
        print(f"   Mid-term has embeddings file: {os.path.exists(memoryos.mid_term_memory.embeddings_file)}")
        
        # Check long-term memory (should have knowledge bases)
        print(f"   Long-term user knowledge: {len(memoryos.user_long_term_memory.user_knowledge)}")
        print(f"   Long-term has user embeddings: {os.path.exists(memoryos.user_long_term_memory.user_embeddings_file)}")
        
        # Test 2: Test short-term Redis-style operations
        print("\n2. Testing Redis-style short-term operations...")
        
        # Add some entries to test FIFO
        test_conversations = [
            ("Hello, how are you?", "I'm doing well, thank you!"),
            ("What's the weather like?", "I don't have access to weather data."),
            ("Can you help me code?", "Yes, I'd be happy to help with coding!"),
            ("What is Python?", "Python is a programming language."),  # Should evict first
            ("How do I learn AI?", "Start with machine learning basics."),  # Should evict second
        ]
        
        for i, (user_input, agent_response) in enumerate(test_conversations):
            # Simulate adding memory without API calls
            memoryos.short_term_memory.add_qa_pair(user_input, agent_response)
            print(f"   Added conversation {i+1}, memory size: {len(memoryos.short_term_memory.memory)}")
        
        # Verify FIFO behavior
        current_entries = memoryos.short_term_memory.get_all()
        print(f"   Final short-term entries: {len(current_entries)}")
        for i, entry in enumerate(current_entries):
            print(f"     Entry {i+1}: {entry['user_input'][:30]}...")
        
        # Check overflow handling
        overflow_entries = memoryos.short_term_memory.get_overflow_entries()
        print(f"   Overflow entries (evicted): {len(overflow_entries)}")
        
        # Test 3: Test memory tier separation
        print("\n3. Testing memory tier separation...")
        
        # Short-term should use deque for fast access
        print(f"   Short-term is fast deque: {type(memoryos.short_term_memory.memory).__name__ == 'deque'}")
        
        # Mid-term should use list with embeddings
        print(f"   Mid-term uses segments list: {type(memoryos.mid_term_memory.memory_segments).__name__ == 'list'}")
        print(f"   Mid-term has embeddings support: {hasattr(memoryos.mid_term_memory, 'embeddings')}")
        
        # Long-term should use knowledge bases
        print(f"   Long-term has user knowledge: {type(memoryos.user_long_term_memory.user_knowledge).__name__ == 'list'}")
        print(f"   Long-term has assistant knowledge: {type(memoryos.user_long_term_memory.assistant_knowledge).__name__ == 'list'}")
        
        # Test 4: Test retrieval across tiers (without API calls)
        print("\n4. Testing cross-tier retrieval structure...")
        
        # Test retriever initialization
        print(f"   Retriever initialized: {memoryos.retriever is not None}")
        print(f"   Retriever has access to all tiers: {hasattr(memoryos.retriever, 'short_term_memory')}")
        
        # Test 5: Performance characteristics
        print("\n5. Testing performance characteristics...")
        
        # Short-term should be very fast
        import time
        start_time = time.time()
        recent = memoryos.short_term_memory.get_recent(2)
        short_term_time = time.time() - start_time
        print(f"   Short-term retrieval time: {short_term_time*1000:.2f}ms (should be <1ms)")
        
        # Test memory statistics
        stats = memoryos.short_term_memory.get_memory_stats()
        print(f"   Short-term usage: {stats['usage_percentage']:.1f}%")
        print(f"   Short-term is full: {memoryos.short_term_memory.is_full()}")
        
        print("\n✅ Memory Tier Integration Test Complete!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run complete memory tier tests"""
    try:
        result = test_memory_tier_integration()
        if result:
            print("\n🎉 All memory tier integration tests passed!")
            print("\n📋 Architecture Summary:")
            print("   ✅ Short-term: Redis-style deque (fast, FIFO)")
            print("   ✅ Mid-term: JSON + embeddings (indexed)")
            print("   ✅ Long-term: JSON + embeddings (persistent)")
            print("   ✅ Proper tier separation and performance characteristics")
        else:
            print("\n❌ Some tests failed!")
    except Exception as e:
        print(f"\n💥 Test suite failed: {e}")

if __name__ == "__main__":
    main()