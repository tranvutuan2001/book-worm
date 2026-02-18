"""
Session Manager for Document Analysis

Manages user sessions and document context for concurrent request handling.
Each session stores the document name and other contextual information needed
for document analysis tools.
"""

import threading
import uuid
from typing import Dict, Optional
from contextvars import ContextVar, copy_context
import logging
from contextlib import contextmanager

logger = logging.getLogger(__name__)

# Context variable for storing session ID - this is thread-safe and async-safe
_session_context: ContextVar[Optional[str]] = ContextVar('session_context', default=None)


class SessionManager:
    """
    Manages document analysis sessions using contextvars for proper isolation
    across concurrent requests and async contexts.
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, 'initialized'):
            self.sessions: Dict[str, Dict] = {}
            self._lock = threading.RLock()
            self.initialized = True
            logger.debug("SessionManager initialized")
    
    def create_session(self, document_name: str) -> str:
        """
        Create a new session with document context
        
        Args:
            document_name: The name of the document to associate with this session
            
        Returns:
            str: Unique session ID
        """
        session_id = str(uuid.uuid4())
        
        with self._lock:
            self.sessions[session_id] = {
                'document_name': document_name,
                'created_at': None  # Could add timestamp if needed
            }
        
        logger.debug(f"Created session {session_id} for document: {document_name}")
        return session_id
    
    def set_current_session(self, session_id: str):
        """
        Set the current session for this execution context
        
        Args:
            session_id: The session ID to set as current
            
        Raises:
            RuntimeError: If session doesn't exist
        """
        with self._lock:
            if session_id not in self.sessions:
                raise RuntimeError(f"Session {session_id} not found.")
        
        _session_context.set(session_id)
        logger.debug(f"Set current session to: {session_id}")
    
    def get_current_session_id(self) -> str:
        """
        Get the current session ID
        
        Returns:
            str: Current session ID
            
        Raises:
            RuntimeError: If no active session
        """
        session_id = _session_context.get()
        if session_id is None:
            raise RuntimeError("No active session. Call set_current_session() first.")
        
        return session_id
    
    def get_current_document_name(self) -> str:
        """
        Get document name from current session
        
        Returns:
            str: Document name associated with current session
            
        Raises:
            RuntimeError: If no active session or session not found
        """
        session_id = self.get_current_session_id()
        
        with self._lock:
            if session_id not in self.sessions:
                raise RuntimeError(f"Session {session_id} not found.")
            
            document_name = self.sessions[session_id]['document_name']
        
        logger.debug(f"Retrieved document name '{document_name}' from session {session_id}")
        return document_name
    
    def cleanup_session(self, session_id: str):
        """
        Clean up session when request is done
        
        Args:
            session_id: The session ID to clean up
        """
        with self._lock:
            if session_id in self.sessions:
                document_name = self.sessions[session_id].get('document_name', 'unknown')
                del self.sessions[session_id]
                logger.debug(f"Cleaned up session {session_id} for document: {document_name}")
            else:
                logger.warning(f"Attempted to cleanup non-existent session: {session_id}")
    
    @contextmanager
    def session_context(self, document_name: str):
        """
        Context manager that creates a session and ensures it's properly cleaned up.
        Also copies the current context to ensure session variables are available
        in any spawned tasks or threads.
        
        Args:
            document_name: The document name to associate with this session
            
        Yields:
            str: The session ID
        """
        session_id = self.create_session(document_name)
        self.set_current_session(session_id)
        
        try:
            logger.debug(f"Entering session context for session {session_id}")
            yield session_id
        finally:
            self.cleanup_session(session_id)
            logger.debug(f"Exited session context for session {session_id}")

# Global session manager instance
session_manager = SessionManager()