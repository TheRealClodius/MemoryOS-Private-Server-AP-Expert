#!/usr/bin/env python3
"""
Simple test for MemoryOS MCP Server
Direct import testing without client/server communication
"""

import json
import sys
import os
from pathlib import Path

# Add the current directory to sys.path to import memoryos
sys.path.insert(0, str(Path(__file__).parent))

def test_memoryos_import():
    """Test that MemoryOS can be imported and initialized"""
    print("🧠 Testing MemoryOS import and initialization...")
    
    try:
        from memoryos import Memoryos
        print("✅ MemoryOS imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Failed to import MemoryOS: {e}")
        return False

def test_mcp_server_import():
    """Test that MCP server can be imported"""
    print("🔧 Testing MCP server imports...")
    
    try:
        import mcp_server
        print("✅ MCP server imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Failed to import MCP server: {e}")
        return False

def test_config_loading():
    """Test configuration loading"""
    print("⚙️ Testing configuration loading...")
    
    try:
        from mcp_server import load_config
        config = load_config()
        print("✅ Configuration loaded successfully")
        print(f"   User ID: {config.get('user_id', 'Not set')}")
        print(f"   OpenAI API Key: {'Set' if config.get('openai_api_key') else 'Not set'}")
        print(f"   Data Path: {config.get('data_storage_path', 'Not set')}")
        return True
    except Exception as e:
        print(f"❌ Failed to load configuration: {e}")
        return False

def test_memoryos_initialization():
    """Test MemoryOS initialization with test config"""
    print("🚀 Testing MemoryOS initialization...")
    
    try:
        from memoryos import Memoryos
        
        # Test configuration (will fail without API key, but that's expected)
        test_config = {
            "user_id": "test_user",
            "openai_api_key": "test_key_placeholder",  # Will fail but tests structure
            "data_storage_path": "./test_memoryos_data",
            "assistant_id": "test_assistant",
            "short_term_capacity": 5,
            "mid_term_capacity": 100,
            "long_term_knowledge_capacity": 50,
            "retrieval_queue_capacity": 3,
            "mid_term_heat_threshold": 3.0,
            "llm_model": "gpt-4o-mini",
            "embedding_model": "text-embedding-3-small"
        }
        
        # Try to initialize (will fail on API calls but should create structure)
        memory_os = Memoryos(**test_config)
        print("✅ MemoryOS instance created successfully")
        print(f"   User ID: {memory_os.user_id}")
        print(f"   Assistant ID: {memory_os.assistant_id}")
        print(f"   LLM Model: {memory_os.llm_model}")
        print(f"   Embedding Model: {memory_os.embedding_model}")
        
        # Test memory components exist
        assert hasattr(memory_os, 'short_term_memory'), "Short-term memory not initialized"
        assert hasattr(memory_os, 'mid_term_memory'), "Mid-term memory not initialized"
        assert hasattr(memory_os, 'user_long_term_memory'), "Long-term memory not initialized"
        assert hasattr(memory_os, 'retriever'), "Retriever not initialized"
        assert hasattr(memory_os, 'updater'), "Updater not initialized"
        
        print("✅ All memory components initialized")
        return True
        
    except Exception as e:
        print(f"❌ Failed to initialize MemoryOS: {e}")
        return False

def test_mcp_tools_structure():
    """Test MCP tool function structures"""
    print("🔧 Testing MCP tool structures...")
    
    try:
        from mcp_server import add_memory, retrieve_memory, get_user_profile
        print("✅ MCP tool functions imported successfully")
        
        # Check function signatures
        import inspect
        
        add_memory_sig = inspect.signature(add_memory)
        print(f"   add_memory parameters: {list(add_memory_sig.parameters.keys())}")
        
        retrieve_memory_sig = inspect.signature(retrieve_memory)
        print(f"   retrieve_memory parameters: {list(retrieve_memory_sig.parameters.keys())}")
        
        get_user_profile_sig = inspect.signature(get_user_profile)
        print(f"   get_user_profile parameters: {list(get_user_profile_sig.parameters.keys())}")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to test MCP tools: {e}")
        return False

def test_pydantic_models():
    """Test Pydantic model structures"""
    print("📋 Testing Pydantic model structures...")
    
    try:
        from mcp_server import (
            MemoryOperationResult, 
            MemoryEntry, 
            KnowledgeEntry, 
            MemoryRetrievalResult, 
            UserProfileResult
        )
        
        # Test model creation
        test_memory_result = MemoryOperationResult(
            status="success",
            message="Test message",
            timestamp="2025-01-01T00:00:00"
        )
        print("✅ MemoryOperationResult model works")
        
        test_memory_entry = MemoryEntry(
            user_input="Test input",
            agent_response="Test response", 
            timestamp="2025-01-01T00:00:00"
        )
        print("✅ MemoryEntry model works")
        
        test_knowledge_entry = KnowledgeEntry(
            knowledge="Test knowledge",
            timestamp="2025-01-01T00:00:00"
        )
        print("✅ KnowledgeEntry model works")
        
        print("✅ All Pydantic models validated successfully")
        return True
        
    except Exception as e:
        print(f"❌ Failed to test Pydantic models: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Starting MemoryOS Component Test Suite")
    print("=" * 50)
    
    tests = [
        ("MemoryOS Import", test_memoryos_import),
        ("MCP Server Import", test_mcp_server_import),
        ("Configuration Loading", test_config_loading),
        ("MemoryOS Initialization", test_memoryos_initialization),
        ("MCP Tools Structure", test_mcp_tools_structure),
        ("Pydantic Models", test_pydantic_models)
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
            print()  # Add spacing between tests
        except Exception as e:
            print(f"❌ Test {test_name} crashed: {e}")
            results[test_name] = False
            print()
    
    # Summary
    print("=" * 50)
    print("📊 Test Results Summary:")
    passed = sum(results.values())
    total = len(results)
    
    for test_name, passed_test in results.items():
        status = "✅ PASS" if passed_test else "❌ FAIL"
        print(f"   {test_name}: {status}")
    
    print(f"\n🏆 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All component tests passed! MemoryOS structure is correct.")
        print("\n📝 Next steps:")
        print("   1. Set OPENAI_API_KEY environment variable")
        print("   2. Run the MCP server: python mcp_server.py")
        print("   3. Connect via MCP client (Claude Desktop, etc.)")
    else:
        print("⚠️ Some tests failed. Check the output above for details.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)