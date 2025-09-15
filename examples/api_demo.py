"""
Demo script for Agent Scrivener API functionality.

This script demonstrates the REST API endpoints and WebSocket real-time updates.
"""

import asyncio
import json
import websockets
from datetime import datetime
from agent_scrivener.api.auth import create_access_token
from agent_scrivener.api.main import app
import uvicorn
import threading
import time
import httpx


class APIDemo:
    """Demo class for testing API functionality."""
    
    def __init__(self):
        self.base_url = "http://localhost:8000"
        self.token = create_access_token("demo_user", "demo", ["read", "write"])
        self.headers = {"Authorization": f"Bearer {self.token}"}
    
    async def test_health_check(self):
        """Test the health check endpoint."""
        print("🔍 Testing health check endpoint...")
        
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{self.base_url}/api/v1/health")
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Health check passed: {data['status']}")
                print(f"   Version: {data['version']}")
                print(f"   Uptime: {data['uptime_seconds']:.1f}s")
            else:
                print(f"❌ Health check failed: {response.status_code}")
    
    async def test_start_research(self):
        """Test starting a research session."""
        print("\n🚀 Starting research session...")
        
        request_data = {
            "query": "What are the latest developments in quantum computing and their potential applications?",
            "max_sources": 8,
            "include_academic": True,
            "include_web": True,
            "priority": "high"
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/api/v1/research",
                json=request_data,
                headers=self.headers
            )
            
            if response.status_code == 200:
                data = response.json()
                session_id = data["session_id"]
                print(f"✅ Research session started: {session_id}")
                print(f"   Status: {data['status']}")
                print(f"   Estimated duration: {data['estimated_duration_minutes']} minutes")
                return session_id
            else:
                print(f"❌ Failed to start research: {response.status_code}")
                print(f"   Error: {response.text}")
                return None
    
    async def test_session_status(self, session_id: str):
        """Test getting session status."""
        print(f"\n📊 Checking session status for {session_id}...")
        
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.base_url}/api/v1/research/{session_id}/status",
                headers=self.headers
            )
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Session status: {data['status']}")
                print(f"   Progress: {data['progress_percentage']:.1f}%")
                if data.get('current_task'):
                    print(f"   Current task: {data['current_task']}")
                if data.get('completed_tasks'):
                    print(f"   Completed tasks: {len(data['completed_tasks'])}")
                return data
            else:
                print(f"❌ Failed to get status: {response.status_code}")
                return None
    
    async def test_websocket_progress(self, session_id: str):
        """Test WebSocket progress updates."""
        print(f"\n🔄 Connecting to WebSocket for session {session_id}...")
        
        ws_url = f"ws://localhost:8000/api/v1/ws/research/{session_id}/progress?token={self.token}"
        
        try:
            async with websockets.connect(ws_url) as websocket:
                print("✅ WebSocket connected")
                
                # Listen for progress updates
                update_count = 0
                while update_count < 10:  # Listen for up to 10 updates
                    try:
                        message = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                        data = json.loads(message)
                        
                        if data["type"] == "status_update":
                            print(f"📈 Progress: {data['progress_percentage']:.1f}% - {data.get('current_task', 'N/A')}")
                        elif data["type"] == "task_completed":
                            print(f"✅ Task completed: {data['task_name']}")
                        elif data["type"] == "session_completed":
                            print(f"🎉 Session completed!")
                            break
                        elif data["type"] == "session_failed":
                            print(f"❌ Session failed: {data['error_message']}")
                            break
                        
                        update_count += 1
                        
                    except asyncio.TimeoutError:
                        print("⏰ No updates received in 5 seconds")
                        break
                
                print("🔌 WebSocket connection closed")
                
        except Exception as e:
            print(f"❌ WebSocket error: {str(e)}")
    
    async def test_get_result(self, session_id: str):
        """Test getting research results."""
        print(f"\n📄 Getting research result for {session_id}...")
        
        # Wait a bit for the session to complete
        await asyncio.sleep(2)
        
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.base_url}/api/v1/research/{session_id}/result",
                headers=self.headers
            )
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Research completed!")
                print(f"   Status: {data['status']}")
                print(f"   Sources used: {data['sources_count']}")
                print(f"   Word count: {data['word_count']}")
                print(f"   Completion time: {data['completion_time_minutes']:.1f} minutes")
                print(f"   Document preview: {data['document_content'][:200]}...")
            else:
                print(f"❌ Failed to get result: {response.status_code}")
                if response.status_code == 400:
                    print("   (Session may not be completed yet)")
    
    async def test_list_sessions(self):
        """Test listing user sessions."""
        print("\n📋 Listing user sessions...")
        
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.base_url}/api/v1/research?page=1&page_size=5",
                headers=self.headers
            )
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Found {data['total_count']} sessions")
                for session in data['sessions']:
                    print(f"   - {session['session_id']}: {session['status']} ({session['progress_percentage']:.1f}%)")
            else:
                print(f"❌ Failed to list sessions: {response.status_code}")
    
    async def run_demo(self):
        """Run the complete API demo."""
        print("🎬 Starting Agent Scrivener API Demo")
        print("=" * 50)
        
        # Test health check
        await self.test_health_check()
        
        # Start research session
        session_id = await self.test_start_research()
        if not session_id:
            return
        
        # Test WebSocket progress updates (run in background)
        websocket_task = asyncio.create_task(self.test_websocket_progress(session_id))
        
        # Monitor session status
        for i in range(5):
            await asyncio.sleep(3)
            await self.test_session_status(session_id)
        
        # Wait for WebSocket task to complete
        await websocket_task
        
        # Get final result
        await self.test_get_result(session_id)
        
        # List all sessions
        await self.test_list_sessions()
        
        print("\n🎉 Demo completed!")


def start_server():
    """Start the FastAPI server in a separate thread."""
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")


async def main():
    """Main demo function."""
    # Start server in background thread
    server_thread = threading.Thread(target=start_server, daemon=True)
    server_thread.start()
    
    # Wait for server to start
    print("⏳ Starting server...")
    await asyncio.sleep(3)
    
    # Run demo
    demo = APIDemo()
    await demo.run_demo()


if __name__ == "__main__":
    asyncio.run(main())