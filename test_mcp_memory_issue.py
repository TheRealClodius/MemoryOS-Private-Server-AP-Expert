#!/usr/bin/env python3
"""
Test to identify the exact issue with MCP server memory retrieval
"""

import sys
import os
from pathlib import Path

# Add the current directory to sys.path to import memoryos
sys.path.insert(0, str(Path(__file__).parent))

from memoryos.short_term import ShortTermMemory
from memoryos.retriever import MemoryRetriever
from memoryos.mid_term import MidTermMemory
from memoryos.long_term import LongTermMemory

def test_retriever_context():
    """Test the retriever context method that's used by MCP server"""
    print("🧪 Testing Memory Retriever Context Method")
    print("=" * 50)
    
    # Initialize memory components (same as in MemoryOS)
    short_term = ShortTermMemory(
        user_id="demo_user",
        data_path="./memoryos_data",
        capacity=10
    )
    
    mid_term = MidTermMemory(
        user_id="demo_user",
        data_path="./memoryos_data",
        capacity=2000,
        heat_threshold=5.0
    )
    
    long_term = LongTermMemory(
        user_id="demo_user",
        assistant_id="mcp_assistant",
        data_path="./memoryos_data",
        knowledge_capacity=100
    )
    
    # Initialize retriever
    retriever = MemoryRetriever(
        short_term_memory=short_term,
        mid_term_memory=mid_term,
        long_term_memory=long_term,
        queue_capacity=7
    )
    
    print(f"Short-term memories: {len(short_term.get_all())}")
    print(f"Mid-term segments: {len(mid_term.memory_segments)}")
    print(f"Long-term user knowledge: {len(long_term.user_knowledge)}")
    print(f"Long-term assistant knowledge: {len(long_term.assistant_knowledge)}")
    
    # Test retrieve_context without embedding (this is what might be failing)
    print("\n1. Testing retrieve_context without embedding...")
    context = retriever.retrieve_context(
        user_query="What do you know about France?",
        user_id="demo_user",
        query_embedding=None  # This is likely the issue
    )
    
    print(f"   Short-term context: {len(context.get('short_term_memory', []))}")
    print(f"   Mid-term context: {len(context.get('retrieved_pages', []))}")
    print(f"   User knowledge: {len(context.get('retrieved_user_knowledge', []))}")
    print(f"   Assistant knowledge: {len(context.get('retrieved_assistant_knowledge', []))}")
    
    # Show actual short-term context
    st_context = context.get('short_term_memory', [])
    print(f"\n   Short-term memories found:")
    for i, memory in enumerate(st_context):
        print(f"     {i+1}. {memory.get('user_input', '')[:50]}...")
    
    return len(st_context) > 0

def test_short_term_context_method():
    """Test the _get_short_term_context method directly"""
    print("\n🧪 Testing Short-term Context Method")
    print("=" * 50)
    
    # Initialize memory components
    short_term = ShortTermMemory(
        user_id="demo_user",
        data_path="./memoryos_data",
        capacity=10
    )
    
    mid_term = MidTermMemory(
        user_id="demo_user",
        data_path="./memoryos_data",
        capacity=2000,
        heat_threshold=5.0
    )
    
    long_term = LongTermMemory(
        user_id="demo_user",
        assistant_id="mcp_assistant",
        data_path="./memoryos_data",
        knowledge_capacity=100
    )
    
    retriever = MemoryRetriever(
        short_term_memory=short_term,
        mid_term_memory=mid_term,
        long_term_memory=long_term,
        queue_capacity=7
    )
    
    # Test _get_short_term_context directly
    print("Testing _get_short_term_context method...")
    st_context = retriever._get_short_term_context("What do you know about France?")
    print(f"   Found {len(st_context)} short-term memories")
    
    for i, memory in enumerate(st_context):
        print(f"     {i+1}. {memory.get('user_input', '')[:50]}...")
    
    return len(st_context) > 0

def test_mcp_server_flow():
    """Test the exact flow used by MCP server"""
    print("\n🧪 Testing MCP Server Flow")
    print("=" * 50)
    
    # Import and test the MCP server components
    try:
        from mcp_server import memoryos_instance, init_memoryos, load_config
        
        print("1. Testing config loading...")
        # Create a minimal config for testing
        test_config = {
            "user_id": "demo_user",
            "openai_api_key": "sk-test-key",  # Dummy key for testing
            "openai_base_url": "https://api.openai.com/v1",
            "data_storage_path": "./memoryos_data",
            "assistant_id": "mcp_assistant",
            "llm_model": "gpt-4o-mini",
            "embedding_model": "text-embedding-3-small",
            "short_term_capacity": 10,
            "mid_term_capacity": 2000,
            "long_term_knowledge_capacity": 100,
            "retrieval_queue_capacity": 7,
            "mid_term_heat_threshold": 5.0
        }
        
        print("2. Testing MemoryOS initialization...")
        # Initialize MemoryOS instance
        test_memoryos = init_memoryos(test_config)
        
        print("3. Testing retriever context...")
        # Test the retriever without embedding (this is what the MCP server does)
        context = test_memoryos.retriever.retrieve_context(
            user_query="What do you know about France?",
            user_id="demo_user",
            query_embedding=None
        )
        
        print(f"   Short-term memories: {len(context.get('short_term_memory', []))}")
        
        # Show the actual memories
        st_memories = context.get('short_term_memory', [])
        for i, memory in enumerate(st_memories):
            print(f"     {i+1}. {memory.get('user_input', '')[:50]}...")
        
        return len(st_memories) > 0
        
    except Exception as e:
        print(f"❌ MCP server flow test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("🚀 MCP Memory Issue Diagnostic Test")
    print("=" * 50)
    
    results = []
    
    # Test 1: Retriever context
    try:
        results.append(("Retriever Context", test_retriever_context()))
    except Exception as e:
        print(f"❌ Retriever context test failed: {e}")
        results.append(("Retriever Context", False))
    
    # Test 2: Short-term context method
    try:
        results.append(("Short-term Context Method", test_short_term_context_method()))
    except Exception as e:
        print(f"❌ Short-term context method test failed: {e}")
        results.append(("Short-term Context Method", False))
    
    # Test 3: MCP server flow
    try:
        results.append(("MCP Server Flow", test_mcp_server_flow()))
    except Exception as e:
        print(f"❌ MCP server flow test failed: {e}")
        results.append(("MCP Server Flow", False))
    
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
        print("🎉 All tests passed! MCP server should work correctly.")
        return True
    else:
        print("⚠️ Some tests failed. This identifies the exact issue.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 