#!/usr/bin/env python3
"""
Test client for MemoryOS MCP Server
Tests all three main tools: add_memory, retrieve_memory, get_user_profile
"""

import asyncio
import json
import sys
from typing import Dict, Any

try:
    from mcp.client.session import ClientSession
    from mcp.client.stdio import StdioServerParameters, stdio_client
    from mcp.types import CallToolRequest
except ImportError as e:
    print(f"ERROR: Failed to import MCP client. Please install: pip install mcp", file=sys.stderr)
    print(f"Import error: {e}", file=sys.stderr)
    sys.exit(1)

class MemoryOSTestClient:
    """Test client for MemoryOS MCP Server"""
    
    def __init__(self):
        self.session: ClientSession = None
    
    async def connect(self):
        """Connect to the MemoryOS MCP server"""
        try:
            # Start the server process
            import subprocess
            import os
            
            # Get the directory of this script
            script_dir = os.path.dirname(os.path.abspath(__file__))
            server_script = os.path.join(script_dir, "mcp_server.py")
            
            # Create server parameters
            server_params = StdioServerParameters(
                command=sys.executable,
                args=[server_script]
            )
            
            # Create stdio client context manager
            stdio_context = stdio_client(server_params)
            
            # Get the streams
            read_stream, write_stream = await stdio_context.__aenter__()
            
            # Initialize session
            self.session = ClientSession(read_stream, write_stream)
            
            # Initialize the session
            init_result = await self.session.initialize()
            print(f"✅ Connected to MemoryOS MCP Server")
            print(f"   Server: {init_result.server_info.name} v{init_result.server_info.version}")
            
            # List available tools
            tools_result = await self.session.list_tools()
            print(f"📋 Available tools: {len(tools_result.tools)}")
            for tool in tools_result.tools:
                print(f"   - {tool.name}: {tool.description}")
            
            return stdio_context
            
        except Exception as e:
            print(f"❌ Failed to connect to server: {e}")
            raise
    
    async def test_add_memory(self) -> bool:
        """Test the add_memory tool"""
        print("\n🧠 Testing add_memory tool...")
        
        try:
            # Test adding a memory
            result = await self.session.call_tool(
                CallToolRequest(
                    method="call_tool",
                    params={
                        "name": "add_memory",
                        "arguments": {
                            "user_input": "Hi! I'm Alice, a software engineer working at TechCorp in San Francisco. I specialize in machine learning and AI systems.",
                            "agent_response": "Hello Alice! It's great to meet you. Machine learning and AI systems are fascinating fields. What kind of projects are you currently working on at TechCorp?",
                            "meta_data": {
                                "test_session": True,
                                "conversation_id": "test_001"
                            }
                        }
                    }
                )
            )
            
            if result.is_error:
                print(f"❌ Error: {result.error}")
                return False
            
            # Parse result
            memory_result = json.loads(result.content[0].text)
            print(f"✅ Memory added successfully")
            print(f"   Status: {memory_result['status']}")
            print(f"   Message: {memory_result['message']}")
            print(f"   Details: {memory_result.get('details', {})}")
            
            # Add another memory
            result2 = await self.session.call_tool(
                CallToolRequest(
                    method="call_tool",
                    params={
                        "name": "add_memory",
                        "arguments": {
                            "user_input": "I'm working on a recommendation system that uses deep learning to personalize content for users. It's challenging but exciting!",
                            "agent_response": "That sounds like a really interesting project! Recommendation systems are crucial for user experience. What kind of deep learning architecture are you using? Are you working with collaborative filtering, content-based approaches, or hybrid methods?"
                        }
                    }
                )
            )
            
            if result2.is_error:
                print(f"❌ Error adding second memory: {result2.error}")
                return False
            
            memory_result2 = json.loads(result2.content[0].text)
            print(f"✅ Second memory added: {memory_result2['status']}")
            
            return True
            
        except Exception as e:
            print(f"❌ Test failed: {e}")
            return False
    
    async def test_retrieve_memory(self) -> bool:
        """Test the retrieve_memory tool"""
        print("\n🔍 Testing retrieve_memory tool...")
        
        try:
            # Test retrieving memories
            result = await self.session.call_tool(
                CallToolRequest(
                    method="call_tool",
                    params={
                        "name": "retrieve_memory",
                        "arguments": {
                            "query": "What do you know about Alice's work and projects?",
                            "relationship_with_user": "assistant",
                            "max_results": 5
                        }
                    }
                )
            )
            
            if result.is_error:
                print(f"❌ Error: {result.error}")
                return False
            
            # Parse result
            retrieval_result = json.loads(result.content[0].text)
            print(f"✅ Memory retrieval successful")
            print(f"   Status: {retrieval_result['status']}")
            print(f"   Query: {retrieval_result['query']}")
            print(f"   User Profile: {retrieval_result['user_profile']}")
            print(f"   Short-term memories: {len(retrieval_result['short_term_memory'])}")
            print(f"   Retrieved pages: {len(retrieval_result['retrieved_pages'])}")
            print(f"   User knowledge: {len(retrieval_result['retrieved_user_knowledge'])}")
            print(f"   Assistant knowledge: {len(retrieval_result['retrieved_assistant_knowledge'])}")
            
            # Display some short-term memories
            if retrieval_result['short_term_memory']:
                print("   Recent conversations:")
                for i, memory in enumerate(retrieval_result['short_term_memory'][:2]):
                    print(f"     {i+1}. User: {memory['user_input'][:100]}...")
                    print(f"        Assistant: {memory['agent_response'][:100]}...")
            
            return True
            
        except Exception as e:
            print(f"❌ Test failed: {e}")
            return False
    
    async def test_get_user_profile(self) -> bool:
        """Test the get_user_profile tool"""
        print("\n👤 Testing get_user_profile tool...")
        
        try:
            # Test getting user profile
            result = await self.session.call_tool(
                CallToolRequest(
                    method="call_tool",
                    params={
                        "name": "get_user_profile",
                        "arguments": {
                            "include_knowledge": True,
                            "include_assistant_knowledge": True
                        }
                    }
                )
            )
            
            if result.is_error:
                print(f"❌ Error: {result.error}")
                return False
            
            # Parse result
            profile_result = json.loads(result.content[0].text)
            print(f"✅ User profile retrieval successful")
            print(f"   Status: {profile_result['status']}")
            print(f"   User ID: {profile_result['user_id']}")
            print(f"   Assistant ID: {profile_result['assistant_id']}")
            print(f"   Profile: {profile_result['user_profile']}")
            
            if profile_result.get('user_knowledge'):
                print(f"   User knowledge entries: {profile_result['user_knowledge_count']}")
            
            if profile_result.get('assistant_knowledge'):
                print(f"   Assistant knowledge entries: {profile_result['assistant_knowledge_count']}")
            
            return True
            
        except Exception as e:
            print(f"❌ Test failed: {e}")
            return False
    
    async def test_error_handling(self) -> bool:
        """Test error handling"""
        print("\n⚠️ Testing error handling...")
        
        try:
            # Test with empty user input
            result = await self.session.call_tool(
                CallToolRequest(
                    method="call_tool",
                    params={
                        "name": "add_memory",
                        "arguments": {
                            "user_input": "",
                            "agent_response": "This should fail"
                        }
                    }
                )
            )
            
            if result.is_error:
                print(f"✅ Server properly rejected invalid request")
                return True
            
            memory_result = json.loads(result.content[0].text)
            if memory_result['status'] == 'error':
                print(f"✅ Error properly handled: {memory_result['message']}")
                return True
            else:
                print(f"❌ Expected error but got success")
                return False
            
        except Exception as e:
            print(f"❌ Error handling test failed: {e}")
            return False
    
    async def run_comprehensive_test(self):
        """Run comprehensive test suite"""
        print("🚀 Starting MemoryOS MCP Server Test Suite")
        print("=" * 50)
        
        try:
            # Connect to server
            server_process = await self.connect()
            
            # Run tests
            tests = [
                ("add_memory", self.test_add_memory),
                ("retrieve_memory", self.test_retrieve_memory),
                ("get_user_profile", self.test_get_user_profile),
                ("error_handling", self.test_error_handling)
            ]
            
            results = {}
            for test_name, test_func in tests:
                try:
                    results[test_name] = await test_func()
                except Exception as e:
                    print(f"❌ Test {test_name} crashed: {e}")
                    results[test_name] = False
            
            # Summary
            print("\n" + "=" * 50)
            print("📊 Test Results Summary:")
            passed = sum(results.values())
            total = len(results)
            
            for test_name, passed_test in results.items():
                status = "✅ PASS" if passed_test else "❌ FAIL"
                print(f"   {test_name}: {status}")
            
            print(f"\n🏆 Overall: {passed}/{total} tests passed")
            
            if passed == total:
                print("🎉 All tests passed! MemoryOS MCP Server is working correctly.")
            else:
                print("⚠️ Some tests failed. Check the output above for details.")
            
            # Cleanup
            print("\n🧹 Cleaning up...")
            server_process.terminate()
            await asyncio.sleep(1)
            
        except Exception as e:
            print(f"❌ Test suite failed: {e}")
            import traceback
            traceback.print_exc()

async def main():
    """Main test function"""
    client = MemoryOSTestClient()
    await client.run_comprehensive_test()

if __name__ == "__main__":
    asyncio.run(main())
