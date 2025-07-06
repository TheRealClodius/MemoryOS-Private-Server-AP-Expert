#!/usr/bin/env python3
"""
Test to verify if embedding generation is causing the retrieval issue
"""

import sys
import os
from pathlib import Path
import numpy as np

# Add the current directory to sys.path to import memoryos
sys.path.insert(0, str(Path(__file__).parent))

def test_embedding_generation_without_api_key():
    """Test what happens when we try to generate embeddings without API key"""
    print("🧪 Testing Embedding Generation Without API Key")
    print("=" * 50)
    
    try:
        from mcp_server import init_memoryos
        
        # Create config with dummy API key
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
        
        print("1. Initializing MemoryOS with invalid API key...")
        memoryos_instance = init_memoryos(test_config)
        
        print("2. Testing embedding generation...")
        try:
            embedding = memoryos_instance._generate_embedding("What is the capital of France?")
            print(f"   Embedding generated: shape={embedding.shape}")
            print(f"   Embedding is all zeros: {np.all(embedding == 0)}")
            return embedding
        except Exception as e:
            print(f"   ❌ Embedding generation failed: {e}")
            return None
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return None

def test_retrieval_with_failed_embedding():
    """Test retrieval when embedding generation fails"""
    print("\n🧪 Testing Retrieval With Failed Embedding")
    print("=" * 50)
    
    try:
        from mcp_server import init_memoryos
        
        # Create config with dummy API key
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
        
        print("1. Initializing MemoryOS with invalid API key...")
        memoryos_instance = init_memoryos(test_config)
        
        print("2. Testing retrieval with failed embedding generation...")
        try:
            # Try to generate embedding (will fail)
            query_embedding = memoryos_instance._generate_embedding("What is the capital of France?")
            print(f"   Generated embedding: shape={query_embedding.shape}")
            
            # Try retrieval with this embedding
            context = memoryos_instance.retriever.retrieve_context(
                user_query="What is the capital of France?",
                user_id="demo_user",
                query_embedding=query_embedding
            )
            
            print(f"   Short-term memories retrieved: {len(context.get('short_term_memory', []))}")
            return len(context.get('short_term_memory', [])) > 0
            
        except Exception as e:
            print(f"   ❌ Retrieval failed: {e}")
            return False
            
    except Exception as e:
        print(f"❌ Test setup failed: {e}")
        return False

def test_retrieval_without_embedding():
    """Test retrieval without embedding generation"""
    print("\n🧪 Testing Retrieval Without Embedding Generation")
    print("=" * 50)
    
    try:
        from mcp_server import init_memoryos
        
        # Create config with dummy API key
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
        
        print("1. Initializing MemoryOS with invalid API key...")
        memoryos_instance = init_memoryos(test_config)
        
        print("2. Testing retrieval WITHOUT embedding generation...")
        try:
            # Skip embedding generation and pass None
            context = memoryos_instance.retriever.retrieve_context(
                user_query="What is the capital of France?",
                user_id="demo_user",
                query_embedding=None  # No embedding
            )
            
            print(f"   Short-term memories retrieved: {len(context.get('short_term_memory', []))}")
            
            # Show the memories
            for i, memory in enumerate(context.get('short_term_memory', [])):
                print(f"     {i+1}. {memory.get('user_input', '')[:50]}...")
            
            return len(context.get('short_term_memory', [])) > 0
            
        except Exception as e:
            print(f"   ❌ Retrieval failed: {e}")
            import traceback
            traceback.print_exc()
            return False
            
    except Exception as e:
        print(f"❌ Test setup failed: {e}")
        return False

def main():
    print("🚀 Embedding Issue Diagnostic Test")
    print("=" * 50)
    
    results = []
    
    # Test 1: Embedding generation
    try:
        embedding = test_embedding_generation_without_api_key()
        results.append(("Embedding Generation", embedding is not None))
    except Exception as e:
        print(f"❌ Embedding generation test crashed: {e}")
        results.append(("Embedding Generation", False))
    
    # Test 2: Retrieval with failed embedding
    try:
        results.append(("Retrieval with Failed Embedding", test_retrieval_with_failed_embedding()))
    except Exception as e:
        print(f"❌ Retrieval with failed embedding test crashed: {e}")
        results.append(("Retrieval with Failed Embedding", False))
    
    # Test 3: Retrieval without embedding
    try:
        results.append(("Retrieval without Embedding", test_retrieval_without_embedding()))
    except Exception as e:
        print(f"❌ Retrieval without embedding test crashed: {e}")
        results.append(("Retrieval without Embedding", False))
    
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
    
    return passed == len(results)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 