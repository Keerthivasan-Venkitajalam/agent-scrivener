#!/usr/bin/env python3
"""
Quick test script to verify MCP servers are working.
"""

import subprocess
import sys
import time

def test_server(server_path, server_name):
    """Test if a server can start without errors."""
    print(f"\n{'='*60}")
    print(f"Testing {server_name}...")
    print(f"{'='*60}")
    
    try:
        # Start the server process
        process = subprocess.Popen(
            [sys.executable, server_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Wait a moment for it to start
        time.sleep(2)
        
        # Check if it's still running
        if process.poll() is None:
            print(f"✅ {server_name} started successfully!")
            print(f"   Process is running (PID: {process.pid})")
            
            # Terminate the process
            process.terminate()
            process.wait(timeout=5)
            print(f"   Process terminated cleanly")
            return True
        else:
            # Process exited
            stdout, stderr = process.communicate()
            print(f"❌ {server_name} failed to start")
            if stdout:
                print(f"   STDOUT: {stdout[:200]}")
            if stderr:
                print(f"   STDERR: {stderr[:200]}")
            return False
            
    except Exception as e:
        print(f"❌ Error testing {server_name}: {str(e)}")
        return False

def main():
    print("MCP Server Test Suite")
    print("=" * 60)
    
    servers = [
        ("mcp_servers/web_research_server.py", "Web Research Server"),
        ("mcp_servers/citation_formatter_server.py", "Citation Formatter Server")
    ]
    
    results = []
    for server_path, server_name in servers:
        result = test_server(server_path, server_name)
        results.append((server_name, result))
    
    # Summary
    print(f"\n{'='*60}")
    print("Test Summary")
    print(f"{'='*60}")
    
    for server_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {server_name}")
    
    all_passed = all(result for _, result in results)
    
    if all_passed:
        print(f"\n🎉 All servers are working correctly!")
        print(f"\nNext steps:")
        print(f"1. Restart Kiro to load the MCP servers")
        print(f"2. Check MCP logs for connection status")
        print(f"3. Test the tools in your agents")
        return 0
    else:
        print(f"\n⚠️  Some servers failed to start")
        print(f"Check the error messages above for details")
        return 1

if __name__ == "__main__":
    sys.exit(main())
