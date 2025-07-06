#!/usr/bin/env python3
"""
Test that simulates the exact MCP tool calls to identify the issue
"""

import sys
import os
from pathlib import Path
import asyncio

# Add the current directory to sys.path to import memoryos
sys.path.insert(0, str(Path(__file__).parent))

async def test_mcp_add_memory_tool():
    """Test the add_memory MCP tool directly"""
    print("🧪 Testing MCP add_memory Tool")
    print("=" * 50)
    
    try:
        # Import MCP server components
        from mcp_server import add_memory, memoryos_instance, init_memoryos
        
        # Create test config
        test_config = {
            "user_id": "demo_user",
            "openai_api_key": "sk-invalid-key",  # Invalid key
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
        
        # Initialize MemoryOS
        print("1. Initializing MemoryOS...")
        # Import and set the global instance
        import mcp_server
        mcp_server.memoryos_instance = init_memoryos(test_config)
        
        # Test add_memory tool
        print("2. Testing add_memory tool...")
        result = await add_memory(
            user_input="What is the weather like today in Paris?",
            agent_response="I don't have access to real-time weather data, but you can check weather.com or your local weather app for current conditions in Paris."
        )
        
        print(f"   Status: {result.status}")
        print(f"   Message: {result.message}")
        print(f"   Timestamp: {result.timestamp}")
        
        return result.status == "success"
        
    except Exception as e:
        print(f"❌ add_memory tool test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_mcp_retrieve_memory_tool():
    """Test the retrieve_memory MCP tool directly"""
    print("\n🧪 Testing MCP retrieve_memory Tool")
    print("=" * 50)
    
    try:
        # Import MCP server components
        from mcp_server import retrieve_memory, memoryos_instance
        
        if memoryos_instance is None:
            print("❌ MemoryOS instance is None - initialization failed")
            return False
        
        # Test retrieve_memory tool
        print("1. Testing retrieve_memory tool...")
        result = await retrieve_memory(
            query="What do you know about France?",
            max_results=10
        )
        
        print(f"   Status: {result.status}")
        print(f"   Query: {result.query}")
        print(f"   User Profile: {result.user_profile[:100]}...")
        print(f"   Short-term Count: {result.short_term_count}")
        print(f"   Retrieved Pages: {len(result.retrieved_pages)}")
        
        # Show short-term memories
        print(f"\n   Short-term memories found:")
        for i, memory in enumerate(result.short_term_memory):
            print(f"     {i+1}. {memory.user_input[:50]}...")
            print(f"        → {memory.agent_response[:50]}...")
        
        return result.status == "success" and result.short_term_count > 0
        
    except Exception as e:
        print(f"❌ retrieve_memory tool test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_mcp_get_user_profile_tool():
    """Test the get_user_profile MCP tool directly"""
    print("\n🧪 Testing MCP get_user_profile Tool")
    print("=" * 50)
    
    try:
        # Import MCP server components
        from mcp_server import get_user_profile, memoryos_instance
        
        if memoryos_instance is None:
            print("❌ MemoryOS instance is None - initialization failed")
            return False
        
        # Test get_user_profile tool
        print("1. Testing get_user_profile tool...")
        result = await get_user_profile(
            include_knowledge=True,
            include_assistant_knowledge=False
        )
        
        print(f"   Status: {result.status}")
        print(f"   User ID: {result.user_id}")
        print(f"   Assistant ID: {result.assistant_id}")
        print(f"   User Profile: {result.user_profile[:100]}...")
        
        return result.status == "success"
        
    except Exception as e:
        print(f"❌ get_user_profile tool test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_memory_count_debug():
    """Debug the memory count issue specifically"""
    print("\n🧪 Debugging Memory Count Issue")
    print("=" * 50)
    
    try:
        from mcp_server import memoryos_instance
        
        if memoryos_instance is None:
            print("❌ MemoryOS instance is None")
            return False
        
        # Get direct memory stats
        print("1. Getting direct memory statistics...")
        stats = memoryos_instance.get_memory_stats()
        
        print(f"   User ID: {stats.get('user_id')}")
        print(f"   Assistant ID: {stats.get('assistant_id')}")
        
        short_term_stats = stats.get('short_term', {})
        print(f"   Short-term total entries: {short_term_stats.get('total_entries', 'N/A')}")
        print(f"   Short-term usage: {short_term_stats.get('usage_percentage', 'N/A')}%")
        
        # Get direct short-term memory content
        print("\n2. Getting direct short-term memory content...")
        all_memories = memoryos_instance.short_term_memory.get_all()
        print(f"   Direct memory count: {len(all_memories)}")
        
        for i, memory in enumerate(all_memories):
            print(f"     {i+1}. {memory.get('user_input', '')[:50]}...")
        
        # Test retriever directly
        print("\n3. Testing retriever directly...")
        context = memoryos_instance.retriever.retrieve_context(
            user_query="What do you know about France?",
            user_id=memoryos_instance.user_id,
            query_embedding=None
        )
        
        print(f"   Retriever short-term count: {len(context.get('short_term_memory', []))}")
        
        return len(all_memories) > 0
        
    except Exception as e:
        print(f"❌ Memory count debug failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    print("🚀 Direct MCP Tools Test")
    print("=" * 50)
    
    results = []
    
    # Test 1: add_memory tool
    try:
        results.append(("add_memory Tool", await test_mcp_add_memory_tool()))
    except Exception as e:
        print(f"❌ add_memory tool test crashed: {e}")
        results.append(("add_memory Tool", False))
    
    # Test 2: Memory count debug
    try:
        results.append(("Memory Count Debug", await test_memory_count_debug()))
    except Exception as e:
        print(f"❌ Memory count debug crashed: {e}")
        results.append(("Memory Count Debug", False))
    
    # Test 3: retrieve_memory tool
    try:
        results.append(("retrieve_memory Tool", await test_mcp_retrieve_memory_tool()))
    except Exception as e:
        print(f"❌ retrieve_memory tool test crashed: {e}")
        results.append(("retrieve_memory Tool", False))
    
    # Test 4: get_user_profile tool
    try:
        results.append(("get_user_profile Tool", await test_mcp_get_user_profile_tool()))
    except Exception as e:
        print(f"❌ get_user_profile tool test crashed: {e}")
        results.append(("get_user_profile Tool", False))
    
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
        print("🎉 All MCP tools work correctly!")
        print("💡 The issue might be in the MCP server startup or client communication.")
    else:
        print("⚠️ MCP tools have issues. This identifies the exact problem.")
    
    return passed == len(results)

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1) 