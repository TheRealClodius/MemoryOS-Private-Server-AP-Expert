#!/usr/bin/env python3
"""
Test deployment entry points and configuration
"""

import os
import sys
import subprocess
import time
import requests
from pathlib import Path

def test_main_py_import():
    """Test that main.py can be imported and configured properly"""
    print("🔍 Testing main.py import and configuration...")
    
    try:
        # Add current directory to path
        sys.path.insert(0, str(Path(__file__).parent))
        
        # Try importing main
        import main
        print("✅ main.py import successful")
        
        # Check if main function exists
        if hasattr(main, 'main'):
            print("✅ main() function found")
        else:
            print("❌ main() function not found")
            return False
            
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Other error: {e}")
        return False

def test_deploy_server_import():
    """Test that deploy_server.py can be imported properly"""
    print("\n🔍 Testing deploy_server.py import...")
    
    try:
        # Try importing deploy_server
        from deploy_server import app
        print("✅ deploy_server.py import successful")
        print("✅ FastAPI app object accessible")
        
        # Check if app has the required attributes
        if hasattr(app, 'routes'):
            route_count = len(app.routes)
            print(f"✅ FastAPI app has {route_count} routes configured")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Other error: {e}")
        return False

def test_environment_variables():
    """Test environment variable configuration"""
    print("\n🔍 Testing environment variables...")
    
    # Check OpenAI API key
    if os.getenv("OPENAI_API_KEY"):
        print("✅ OPENAI_API_KEY is set")
    else:
        print("❌ OPENAI_API_KEY is not set")
        return False
    
    # Check PORT variable (should default to 5000)
    port = os.getenv("PORT", "5000")
    print(f"✅ PORT configured: {port}")
    
    return True

def test_executable_permissions():
    """Test that scripts have proper executable permissions"""
    print("\n🔍 Testing executable permissions...")
    
    scripts = ["main.py", "deploy_server.py", "run.py"]
    all_executable = True
    
    for script in scripts:
        if os.path.exists(script):
            # Check if file has execute permissions
            stat_info = os.stat(script)
            is_executable = bool(stat_info.st_mode & 0o111)
            
            if is_executable:
                print(f"✅ {script} is executable")
            else:
                print(f"❌ {script} is not executable")
                all_executable = False
        else:
            print(f"❌ {script} not found")
            all_executable = False
    
    return all_executable

def main():
    """Run all deployment tests"""
    print("🚀 MemoryOS Deployment Testing Suite")
    print("=" * 50)
    
    tests = [
        ("Main.py Import", test_main_py_import),
        ("Deploy Server Import", test_deploy_server_import),
        ("Environment Variables", test_environment_variables),
        ("Executable Permissions", test_executable_permissions)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Deployment Test Results:")
    passed = 0
    for test_name, passed_test in results:
        status = "✅ PASS" if passed_test else "❌ FAIL"
        print(f"   {test_name}: {status}")
        if passed_test:
            passed += 1
    
    print(f"\n🏆 Overall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 Deployment configuration is ready!")
        print("\n📋 Recommended deployment commands:")
        print("   For Replit: python main.py")
        print("   For Docker: python main.py")
        print("   For direct: python deploy_server.py")
    else:
        print("⚠️ Deployment has issues that need to be addressed.")
    
    return passed == len(results)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)