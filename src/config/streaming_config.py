# Streaming Configuration for NetRestore RAG System

import os
from typing import Dict, Any

class StreamingConfig:
    """Configuration class for streaming functionality"""
    
    # Default streaming settings
    DEFAULT_ENABLED = True
    DEFAULT_CHUNK_SIZE = 5
    DEFAULT_DELAY = 0.05
    DEFAULT_TIMEOUT = 5
    
    # API endpoints
    DEFAULT_API_URL = "https://mia-propertied-cristopher.ngrok-free.dev/ask"
    DEFAULT_STREAM_URL = "https://mia-propertied-cristopher.ngrok-free.dev/ask-stream"
    
    # UI settings
    SHOW_SOURCES_DEFAULT = True
    AUTO_EXPAND_SOURCES = False
    STREAMING_CURSOR_ENABLED = True
    
    # Performance settings
    MAX_MESSAGE_HISTORY = 100
    CONNECTION_CHECK_INTERVAL = 30
    
    @classmethod
    def get_config(cls) -> Dict[str, Any]:
        """Get current streaming configuration"""
        return {
            "enabled": cls.DEFAULT_ENABLED,
            "chunk_size": cls.DEFAULT_CHUNK_SIZE,
            "delay": cls.DEFAULT_DELAY,
            "timeout": cls.DEFAULT_TIMEOUT,
            "api_url": cls.DEFAULT_API_URL,
            "stream_url": cls.DEFAULT_STREAM_URL,
            "show_sources": cls.SHOW_SOURCES_DEFAULT,
            "auto_expand_sources": cls.AUTO_EXPAND_SOURCES,
            "streaming_cursor": cls.STREAMING_CURSOR_ENABLED,
            "max_message_history": cls.MAX_MESSAGE_HISTORY,
            "connection_check_interval": cls.CONNECTION_CHECK_INTERVAL
        }
    
    @classmethod
    def get_env_config(cls) -> Dict[str, Any]:
        """Get configuration from environment variables"""
        return {
            "enabled": os.getenv("STREAMING_ENABLED", str(cls.DEFAULT_ENABLED)).lower() == "true",
            "chunk_size": int(os.getenv("STREAMING_CHUNK_SIZE", str(cls.DEFAULT_CHUNK_SIZE))),
            "delay": float(os.getenv("STREAMING_DELAY", str(cls.DEFAULT_DELAY))),
            "timeout": int(os.getenv("STREAMING_TIMEOUT", str(cls.DEFAULT_TIMEOUT))),
            "api_url": os.getenv("API_URL", cls.DEFAULT_API_URL),
            "stream_url": os.getenv("STREAM_API_URL", cls.DEFAULT_STREAM_URL),
            "show_sources": os.getenv("SHOW_SOURCES", str(cls.SHOW_SOURCES_DEFAULT)).lower() == "true",
            "auto_expand_sources": os.getenv("AUTO_EXPAND_SOURCES", str(cls.AUTO_EXPAND_SOURCES)).lower() == "true",
            "streaming_cursor": os.getenv("STREAMING_CURSOR", str(cls.STREAMING_CURSOR_ENABLED)).lower() == "true",
            "max_message_history": int(os.getenv("MAX_MESSAGE_HISTORY", str(cls.MAX_MESSAGE_HISTORY))),
            "connection_check_interval": int(os.getenv("CONNECTION_CHECK_INTERVAL", str(cls.CONNECTION_CHECK_INTERVAL)))
        }

def get_streaming_config() -> Dict[str, Any]:
    """Get merged configuration (env vars override defaults)"""
    default_config = StreamingConfig.get_config()
    env_config = StreamingConfig.get_env_config()
    
    # Merge configurations (environment variables take precedence)
    merged_config = default_config.copy()
    merged_config.update(env_config)
    
    return merged_config

# Example usage and documentation
"""
Streaming Configuration Guide:

Environment Variables:
- STREAMING_ENABLED: Enable/disable streaming (true/false)
- STREAMING_CHUNK_SIZE: Number of words per chunk (default: 5)
- STREAMING_DELAY: Delay between chunks in seconds (default: 0.05)
- STREAMING_TIMEOUT: Connection timeout in seconds (default: 5)
- API_URL: Regular API endpoint
- STREAM_API_URL: Streaming API endpoint
- SHOW_SOURCES: Show/hide sources by default (true/false)
- AUTO_EXPAND_SOURCES: Auto-expand sources section (true/false)
- STREAMING_CURSOR: Enable streaming cursor animation (true/false)
- MAX_MESSAGE_HISTORY: Maximum messages to keep in history (default: 100)
- CONNECTION_CHECK_INTERVAL: Connection check interval in seconds (default: 30)

Example:
export STREAMING_ENABLED=true
export STREAMING_CHUNK_SIZE=3
export STREAMING_DELAY=0.1
export SHOW_SOURCES=true
"""
