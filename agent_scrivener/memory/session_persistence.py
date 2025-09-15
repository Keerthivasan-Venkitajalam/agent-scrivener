"""
Session persistence layer for Agent Scrivener.

Handles saving and loading research sessions to/from persistent storage.
"""

import json
import os
import aiofiles
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any
import asyncio

from ..models.core import ResearchSession
from ..utils.logging import get_logger
from ..utils.error_handler import ErrorHandler

logger = get_logger(__name__)


class SessionPersistence:
    """
    Handles persistence of research sessions to local storage.
    
    Provides async file-based storage with JSON serialization for
    session data persistence and recovery.
    """
    
    def __init__(self, storage_path: Optional[str] = None):
        """
        Initialize session persistence.
        
        Args:
            storage_path: Path to storage directory. Defaults to ./data/sessions
        """
        self.storage_path = Path(storage_path or "./data/sessions")
        self.error_handler = ErrorHandler()
        self._lock = asyncio.Lock()
        
    async def initialize(self):
        """Initialize the persistence layer."""
        # Create storage directory if it doesn't exist
        self.storage_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Initialized session persistence at {self.storage_path}")
    
    async def save_session(self, session: ResearchSession) -> bool:
        """
        Save a session to persistent storage.
        
        Args:
            session: The session to save
            
        Returns:
            True if save was successful, False otherwise
        """
        session_file = self.storage_path / f"{session.session_id}.json"
        
        try:
            async with self._lock:
                # Convert session to dict
                session_data = session.model_dump(mode='json')
                
                # Add metadata
                session_data['_metadata'] = {
                    'saved_at': datetime.now().isoformat(),
                    'version': '1.0'
                }
                
                # Write to file atomically
                temp_file = session_file.with_suffix('.tmp')
                async with aiofiles.open(temp_file, 'w', encoding='utf-8') as f:
                    await f.write(json.dumps(session_data, indent=2, ensure_ascii=False))
                
                # Atomic rename
                temp_file.replace(session_file)
                
            logger.debug(f"Saved session {session.session_id} to {session_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving session {session.session_id}: {e}")
            return False
    
    async def load_session(self, session_id: str) -> Optional[ResearchSession]:
        """
        Load a session from persistent storage.
        
        Args:
            session_id: The session identifier
            
        Returns:
            Loaded session if found, None otherwise
        """
        session_file = self.storage_path / f"{session_id}.json"
        
        if not session_file.exists():
            return None
        
        try:
            async with self._lock:
                async with aiofiles.open(session_file, 'r', encoding='utf-8') as f:
                    content = await f.read()
                
                session_data = json.loads(content)
                
                # Remove metadata before parsing
                session_data.pop('_metadata', None)
                
                # Parse session
                session = ResearchSession.model_validate(session_data)
                
            logger.debug(f"Loaded session {session_id} from {session_file}")
            return session
            
        except Exception as e:
            logger.error(f"Error loading session {session_id}: {e}")
            return None
    
    async def delete_session(self, session_id: str) -> bool:
        """
        Delete a session from persistent storage.
        
        Args:
            session_id: The session identifier
            
        Returns:
            True if deletion was successful, False otherwise
        """
        session_file = self.storage_path / f"{session_id}.json"
        
        try:
            async with self._lock:
                if session_file.exists():
                    session_file.unlink()
                    logger.debug(f"Deleted session file {session_file}")
                    return True
                else:
                    logger.debug(f"Session file {session_file} does not exist")
                    return False
                    
        except Exception as e:
            logger.error(f"Error deleting session {session_id}: {e}")
            return False
    
    async def list_sessions(self) -> List[ResearchSession]:
        """
        List all persisted sessions.
        
        Returns:
            List of all persisted sessions
        """
        sessions = []
        
        try:
            async with self._lock:
                # Get all session files
                session_files = list(self.storage_path.glob("*.json"))
                
                # Load each session
                for session_file in session_files:
                    try:
                        async with aiofiles.open(session_file, 'r', encoding='utf-8') as f:
                            content = await f.read()
                        
                        session_data = json.loads(content)
                        session_data.pop('_metadata', None)
                        
                        session = ResearchSession.model_validate(session_data)
                        sessions.append(session)
                        
                    except Exception as e:
                        logger.error(f"Error loading session from {session_file}: {e}")
                        continue
                        
        except Exception as e:
            logger.error(f"Error listing sessions: {e}")
        
        return sessions
    
    async def get_session_metadata(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Get metadata for a session without loading the full session.
        
        Args:
            session_id: The session identifier
            
        Returns:
            Session metadata if found, None otherwise
        """
        session_file = self.storage_path / f"{session_id}.json"
        
        if not session_file.exists():
            return None
        
        try:
            async with self._lock:
                async with aiofiles.open(session_file, 'r', encoding='utf-8') as f:
                    content = await f.read()
                
                session_data = json.loads(content)
                
                # Extract basic metadata
                metadata = {
                    'session_id': session_data.get('session_id'),
                    'original_query': session_data.get('original_query'),
                    'status': session_data.get('status'),
                    'session_state': session_data.get('session_state'),
                    'created_at': session_data.get('created_at'),
                    'updated_at': session_data.get('updated_at'),
                    'file_size': session_file.stat().st_size,
                    'saved_at': session_data.get('_metadata', {}).get('saved_at')
                }
                
                return metadata
                
        except Exception as e:
            logger.error(f"Error getting metadata for session {session_id}: {e}")
            return None
    
    async def cleanup_old_sessions(self, max_age_days: int = 30) -> int:
        """
        Clean up old session files.
        
        Args:
            max_age_days: Maximum age of sessions to keep
            
        Returns:
            Number of sessions cleaned up
        """
        cleaned_count = 0
        cutoff_time = datetime.now().timestamp() - (max_age_days * 24 * 60 * 60)
        
        try:
            async with self._lock:
                session_files = list(self.storage_path.glob("*.json"))
                
                for session_file in session_files:
                    try:
                        # Check file modification time
                        if session_file.stat().st_mtime < cutoff_time:
                            session_file.unlink()
                            cleaned_count += 1
                            logger.debug(f"Cleaned up old session file {session_file}")
                            
                    except Exception as e:
                        logger.error(f"Error cleaning up session file {session_file}: {e}")
                        continue
                        
        except Exception as e:
            logger.error(f"Error during session cleanup: {e}")
        
        if cleaned_count > 0:
            logger.info(f"Cleaned up {cleaned_count} old session files")
        
        return cleaned_count
    
    async def get_storage_stats(self) -> Dict[str, Any]:
        """
        Get storage statistics.
        
        Returns:
            Dictionary with storage statistics
        """
        try:
            async with self._lock:
                session_files = list(self.storage_path.glob("*.json"))
                
                total_files = len(session_files)
                total_size = sum(f.stat().st_size for f in session_files)
                
                # Get oldest and newest files
                oldest_file = None
                newest_file = None
                
                if session_files:
                    oldest_file = min(session_files, key=lambda f: f.stat().st_mtime)
                    newest_file = max(session_files, key=lambda f: f.stat().st_mtime)
                
                return {
                    'storage_path': str(self.storage_path),
                    'total_files': total_files,
                    'total_size_bytes': total_size,
                    'total_size_mb': round(total_size / (1024 * 1024), 2),
                    'oldest_file': {
                        'name': oldest_file.name if oldest_file else None,
                        'modified_at': datetime.fromtimestamp(oldest_file.stat().st_mtime).isoformat() if oldest_file else None
                    },
                    'newest_file': {
                        'name': newest_file.name if newest_file else None,
                        'modified_at': datetime.fromtimestamp(newest_file.stat().st_mtime).isoformat() if newest_file else None
                    }
                }
                
        except Exception as e:
            logger.error(f"Error getting storage stats: {e}")
            return {
                'storage_path': str(self.storage_path),
                'error': str(e)
            }
    
    async def backup_sessions(self, backup_path: str) -> bool:
        """
        Create a backup of all sessions.
        
        Args:
            backup_path: Path to backup directory
            
        Returns:
            True if backup was successful, False otherwise
        """
        backup_dir = Path(backup_path)
        
        try:
            # Create backup directory
            backup_dir.mkdir(parents=True, exist_ok=True)
            
            async with self._lock:
                session_files = list(self.storage_path.glob("*.json"))
                
                for session_file in session_files:
                    backup_file = backup_dir / session_file.name
                    
                    # Copy file
                    async with aiofiles.open(session_file, 'r', encoding='utf-8') as src:
                        content = await src.read()
                    
                    async with aiofiles.open(backup_file, 'w', encoding='utf-8') as dst:
                        await dst.write(content)
                
            logger.info(f"Backed up {len(session_files)} sessions to {backup_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating backup: {e}")
            return False
    
    async def restore_sessions(self, backup_path: str, overwrite: bool = False) -> int:
        """
        Restore sessions from a backup.
        
        Args:
            backup_path: Path to backup directory
            overwrite: Whether to overwrite existing sessions
            
        Returns:
            Number of sessions restored
        """
        backup_dir = Path(backup_path)
        restored_count = 0
        
        if not backup_dir.exists():
            logger.error(f"Backup directory {backup_dir} does not exist")
            return 0
        
        try:
            async with self._lock:
                backup_files = list(backup_dir.glob("*.json"))
                
                for backup_file in backup_files:
                    session_file = self.storage_path / backup_file.name
                    
                    # Skip if file exists and overwrite is False
                    if session_file.exists() and not overwrite:
                        continue
                    
                    # Copy file
                    async with aiofiles.open(backup_file, 'r', encoding='utf-8') as src:
                        content = await src.read()
                    
                    async with aiofiles.open(session_file, 'w', encoding='utf-8') as dst:
                        await dst.write(content)
                    
                    restored_count += 1
                
            logger.info(f"Restored {restored_count} sessions from {backup_dir}")
            return restored_count
            
        except Exception as e:
            logger.error(f"Error restoring from backup: {e}")
            return 0