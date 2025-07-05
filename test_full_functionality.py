#!/usr/bin/env python3
"""
Full functionality test for MemoryOS MCP Server with real OpenAI API integration
"""

import asyncio
import json
import sys
import os
from pathlib import Path

# Add the current directory to sys.path
sys.path.insert(0, str(Path(__file__).parent))

async def test_memoryos_with_api():
    """Test MemoryOS with real OpenAI API calls"""
    print("🧠 Testing MemoryOS with real API integration...")
    
    try:
        from mcp_server import init_memoryos, load_config
        
        # Load configuration with API key
        config = load_config()
        print(f"✅ Configuration loaded with API key: {'Yes' if config.get('openai_api_key') else 'No'}")
        
        # Initialize MemoryOS
        memory_os = init_memoryos(config)
        print(f"✅ MemoryOS initialized for user: {memory_os.user_id}")
        
        # Test adding memories
        print("\n📝 Testing memory addition...")
        
        # Add first memory
        result1 = memory_os.add_memory(
            user_input="Hi! I'm Sarah, a data scientist working on recommendation systems. I love machine learning and hiking.",
            agent_response="Hello Sarah! It's great to meet you. Data science and recommendation systems are fascinating fields. What specific ML techniques do you find most effective for recommendations?"
        )
        print(f"   Memory 1 added: {result1['status']}")
        
        # Add second memory
        result2 = memory_os.add_memory(
            user_input="I primarily use collaborative filtering and deep neural networks. I'm currently working on a hybrid approach that combines both.",
            agent_response="That sounds like a sophisticated approach! Hybrid systems often perform better by leveraging the strengths of different techniques. Are you using any specific frameworks like TensorFlow or PyTorch for the neural network components?"
        )
        print(f"   Memory 2 added: {result2['status']}")
        
        # Add third memory
        result3 = memory_os.add_memory(
            user_input="I'm using PyTorch for the deep learning parts. When I'm not coding, I love hiking in the mountains - it helps me think through complex problems.",
            agent_response="PyTorch is an excellent choice for research and experimentation! And hiking sounds like a wonderful way to clear your mind. There's something about being in nature that often leads to breakthrough insights. Do you have a favorite hiking spot?"
        )
        print(f"   Memory 3 added: {result3['status']}")
        
        # Test memory retrieval
        print("\n🔍 Testing memory retrieval...")
        
        from mcp_server import memoryos_instance
        # Set the global instance for testing
        globals()['memoryos_instance'] = memory_os
        
        # Import and test the MCP tools
        from mcp_server import retrieve_memory, get_user_profile
        
        # Test retrieval about Sarah's work
        retrieval_result = await retrieve_memory(
            query="What does Sarah work on and what are her interests?",
            max_results=5
        )
        
        print(f"   Retrieval status: {retrieval_result.status}")
        print(f"   Short-term memories found: {retrieval_result.short_term_count}")
        print(f"   User profile: {retrieval_result.user_profile[:100]}...")
        
        # Test user profile
        profile_result = await get_user_profile(include_knowledge=True)
        print(f"   Profile status: {profile_result.status}")
        print(f"   User ID: {profile_result.user_id}")
        
        # Test memory statistics
        stats = memory_os.get_memory_stats()
        print(f"\n📊 Memory Statistics:")
        print(f"   Short-term entries: {stats['short_term']['current_size']}")
        print(f"   Mid-term segments: {stats['mid_term']['current_size']}")
        print(f"   Long-term user knowledge: {stats['long_term']['user_knowledge_count']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Full functionality test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_embedding_generation():
    """Test embedding generation specifically"""
    print("\n🔢 Testing embedding generation...")
    
    try:
        from mcp_server import init_memoryos, load_config
        
        config = load_config()
        memory_os = init_memoryos(config)
        
        # Test embedding generation
        test_text = "Machine learning and data science"
        embedding = memory_os._generate_embedding(test_text)
        
        print(f"   Text: '{test_text}'")
        print(f"   Embedding shape: {embedding.shape}")
        print(f"   Embedding type: {type(embedding)}")
        print(f"   Non-zero values: {(embedding != 0).sum()}")
        
        if embedding.shape[0] > 0 and (embedding != 0).sum() > 0:
            print("✅ Embedding generation successful")
            return True
        else:
            print("❌ Embedding appears to be zero vector")
            return False
            
    except Exception as e:
        print(f"❌ Embedding test failed: {e}")
        return False

async def test_llm_integration():
    """Test LLM integration"""
    print("\n🤖 Testing LLM integration...")
    
    try:
        from mcp_server import init_memoryos, load_config
        
        config = load_config()
        memory_os = init_memoryos(config)
        
        # Test LLM call
        test_prompt = "Summarize this in one sentence: Sarah is a data scientist who works on recommendation systems using machine learning."
        response = memory_os._call_llm(test_prompt, max_tokens=100)
        
        print(f"   Prompt: {test_prompt}")
        print(f"   Response: {response}")
        
        if response and len(response.strip()) > 0:
            print("✅ LLM integration successful")
            return True
        else:
            print("❌ LLM returned empty response")
            return False
            
    except Exception as e:
        print(f"❌ LLM test failed: {e}")
        return False

async def main():
    """Run full functionality test suite"""
    print("🚀 MemoryOS Full Functionality Test Suite")
    print("=" * 50)
    
    tests = [
        ("Embedding Generation", test_embedding_generation),
        ("LLM Integration", test_llm_integration),
        ("Complete Memory System", test_memoryos_with_api)
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            print(f"\n🧪 Running {test_name}...")
            results[test_name] = await test_func()
        except Exception as e:
            print(f"❌ Test {test_name} crashed: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Full Functionality Test Results:")
    passed = sum(results.values())
    total = len(results)
    
    for test_name, passed_test in results.items():
        status = "✅ PASS" if passed_test else "❌ FAIL"
        print(f"   {test_name}: {status}")
    
    print(f"\n🏆 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All functionality tests passed!")
        print("\n✅ MemoryOS MCP Server is fully operational:")
        print("   - OpenAI API integration working")
        print("   - Memory system functioning")
        print("   - Ready for production deployment")
    else:
        print("⚠️ Some functionality tests failed. Check API configuration.")
    
    return passed == total

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)