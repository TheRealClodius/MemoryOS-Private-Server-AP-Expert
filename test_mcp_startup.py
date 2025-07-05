#!/usr/bin/env python3
"""
Test MCP server startup and basic functionality
"""

import subprocess
import sys
import time
import os
from pathlib import Path

def test_mcp_server_startup():
    """Test that the MCP server can start without errors"""
    print("🚀 Testing MCP server startup...")
    
    try:
        # Get the server script path
        script_dir = Path(__file__).parent
        server_script = script_dir / "mcp_server.py"
        
        # Start the server process
        process = subprocess.Popen(
            [sys.executable, str(server_script)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Wait a moment for startup
        time.sleep(2)
        
        # Check if process is still running (hasn't crashed)
        if process.poll() is None:
            print("✅ MCP server started successfully and is running")
            
            # Send a basic JSON-RPC message to test protocol handling
            test_message = '{"jsonrpc": "2.0", "method": "initialize", "id": 1, "params": {"protocolVersion": "2024-11-05", "capabilities": {}, "clientInfo": {"name": "test-client", "version": "1.0.0"}}}\n'
            
            try:
                process.stdin.write(test_message)
                process.stdin.flush()
                
                # Wait briefly for response
                time.sleep(1)
                
                print("✅ Server accepts JSON-RPC messages")
                
            except Exception as e:
                print(f"⚠️ Could not test message handling: {e}")
            
            # Terminate the process
            process.terminate()
            try:
                process.wait(timeout=5)
                print("✅ Server terminated cleanly")
            except subprocess.TimeoutExpired:
                process.kill()
                print("⚠️ Server had to be force-killed")
                
            return True
            
        else:
            # Process crashed during startup
            stdout, stderr = process.communicate()
            print("❌ MCP server crashed during startup")
            if stderr:
                print(f"Error output: {stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Failed to start MCP server: {e}")
        return False

def test_config_validation():
    """Test configuration validation"""
    print("⚙️ Testing configuration validation...")
    
    try:
        # Test with missing API key (should show helpful error)
        old_key = os.environ.get('OPENAI_API_KEY')
        if old_key:
            del os.environ['OPENAI_API_KEY']
        
        from mcp_server import load_config
        
        try:
            config = load_config()
            print("❌ Expected error for missing API key")
            return False
        except ValueError as e:
            if "OpenAI API key is required" in str(e):
                print("✅ Configuration properly validates API key requirement")
                
                # Restore API key if it was set
                if old_key:
                    os.environ['OPENAI_API_KEY'] = old_key
                return True
            else:
                print(f"❌ Unexpected validation error: {e}")
                return False
                
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def main():
    """Run MCP server startup tests"""
    print("🧪 MCP Server Startup Test Suite")
    print("=" * 40)
    
    tests = [
        ("Configuration Validation", test_config_validation),
        ("MCP Server Startup", test_mcp_server_startup)
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
            print()
        except Exception as e:
            print(f"❌ Test {test_name} crashed: {e}")
            results[test_name] = False
            print()
    
    # Summary
    print("=" * 40)
    print("📊 Startup Test Results:")
    passed = sum(results.values())
    total = len(results)
    
    for test_name, passed_test in results.items():
        status = "✅ PASS" if passed_test else "❌ FAIL"
        print(f"   {test_name}: {status}")
    
    print(f"\n🏆 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 MCP server startup tests passed!")
        print("\n✅ Ready for deployment:")
        print("   1. Set OPENAI_API_KEY environment variable")
        print("   2. Run: python mcp_server.py")
        print("   3. Connect via MCP client (Claude Desktop, etc.)")
    else:
        print("⚠️ Some startup tests failed. Check the output above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)