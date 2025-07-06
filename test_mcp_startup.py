#!/usr/bin/env python3
"""
Test MCP server startup to verify configuration and initialization
"""

import sys
import os
import subprocess
import time
import signal
from pathlib import Path

def test_server_startup():
    """Test that the MCP server starts up correctly"""
    print("🧪 Testing MCP Server Startup")
    print("=" * 50)
    
    # Create a minimal config for testing
    config_content = '''{
  "user_id": "demo_user",
  "openai_api_key": "sk-test-key-for-startup",
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
}'''
    
    # Write test config
    with open('config_test.json', 'w') as f:
        f.write(config_content)
    
    try:
        print("1. Starting MCP server...")
        
                 # Set up environment variables for the server
        test_env = {
            **os.environ,
            'OPENAI_API_KEY': 'sk-test-key-for-startup',
            'MEMORYOS_USER_ID': 'demo_user',
            'MEMORYOS_ASSISTANT_ID': 'mcp_assistant'
        }
        
        # Start server with timeout
        process = subprocess.Popen(
            [sys.executable, 'mcp_server.py'],
            env=test_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Wait for startup (max 10 seconds)
        startup_output = ""
        startup_errors = ""
        
        try:
            stdout, stderr = process.communicate(timeout=5)
            startup_output = stdout
            startup_errors = stderr
            
            print("   Server started and completed initialization")
            print(f"   Return code: {process.returncode}")
            
        except subprocess.TimeoutExpired:
            # Server is still running - this is expected for MCP servers
            print("   Server is running (timeout expected for MCP servers)")
            process.terminate()
            try:
                stdout, stderr = process.communicate(timeout=2)
                startup_output = stdout
                startup_errors = stderr
            except subprocess.TimeoutExpired:
                process.kill()
                stdout, stderr = process.communicate()
                startup_output = stdout
                startup_errors = stderr
        
        # Analyze startup output
        print("\n2. Analyzing startup output...")
        
        if startup_errors:
            print("   Startup messages:")
            for line in startup_errors.split('\n'):
                if line.strip():
                    print(f"     {line}")
        
        # Check for key initialization messages
        success_indicators = [
            "MemoryOS MCP Server started successfully",
            "MemoryOS initialization verified",
            "User: demo_user"
        ]
        
        error_indicators = [
            "ERROR:",
            "Failed to initialize",
            "Exception:",
            "Traceback"
        ]
        
        found_success = 0
        found_errors = 0
        
        all_output = startup_output + startup_errors
        
        for indicator in success_indicators:
            if indicator in all_output:
                found_success += 1
                print(f"   ✅ Found: {indicator}")
        
        for indicator in error_indicators:
            if indicator in all_output:
                found_errors += 1
                print(f"   ❌ Found: {indicator}")
        
        print(f"\n   Success indicators: {found_success}/{len(success_indicators)}")
        print(f"   Error indicators: {found_errors}")
        
        return found_success >= 2 and found_errors == 0
        
    except Exception as e:
        print(f"❌ Server startup test failed: {e}")
        return False
    
    finally:
        # Clean up test config
        try:
            os.remove('config_test.json')
        except:
            pass

def test_config_loading():
    """Test configuration loading with different scenarios"""
    print("\n🧪 Testing Configuration Loading")
    print("=" * 50)
    
    try:
        from mcp_server import load_config
        
        # Test 1: Load with environment variables
        print("1. Testing environment variable configuration...")
        old_env = {}
        test_env_vars = {
            'MEMORYOS_USER_ID': 'test_user',
            'OPENAI_API_KEY': 'sk-test-key',
            'MEMORYOS_ASSISTANT_ID': 'test_assistant'
        }
        
        # Set environment variables
        for key, value in test_env_vars.items():
            old_env[key] = os.environ.get(key)
            os.environ[key] = value
        
        try:
            config = load_config('nonexistent.json')  # Should fallback to env vars
            print(f"   User ID: {config.get('user_id')}")
            print(f"   Assistant ID: {config.get('assistant_id')}")
            print(f"   Has API key: {'openai_api_key' in config and config['openai_api_key']}")
            
            config_success = (
                config.get('user_id') == 'test_user' and
                config.get('assistant_id') == 'test_assistant' and
                config.get('openai_api_key') == 'sk-test-key'
            )
            
        finally:
            # Restore environment
            for key, old_value in old_env.items():
                if old_value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = old_value
        
        return config_success
        
    except Exception as e:
        print(f"❌ Configuration loading test failed: {e}")
        return False

def main():
    print("🚀 MCP Server Startup Test Suite")
    print("=" * 50)
    
    results = []
    
    # Test 1: Configuration loading
    try:
        results.append(("Configuration Loading", test_config_loading()))
    except Exception as e:
        print(f"❌ Configuration loading test crashed: {e}")
        results.append(("Configuration Loading", False))
    
    # Test 2: Server startup
    try:
        results.append(("Server Startup", test_server_startup()))
    except Exception as e:
        print(f"❌ Server startup test crashed: {e}")
        results.append(("Server Startup", False))
    
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
        print("🎉 MCP server startup is working correctly!")
    else:
        print("⚠️ MCP server startup has issues that need to be addressed.")
    
    return passed == len(results)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)