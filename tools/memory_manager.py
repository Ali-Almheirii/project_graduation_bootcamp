"""
Memory management system for conversations and entity data.

This module provides conversation buffer management, entity memory,
and other memory-related functionality for the agent system.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Dict, List, Optional

import database


class ConversationBuffer:
    """Manages conversation history with a sliding window buffer."""
    
    def __init__(self, max_messages: int = 5):
        self.max_messages = max_messages
    
    def add_message(self, conversation_id: int, sender: str, content: str) -> None:
        """Add a message to the conversation buffer."""
        try:
            # Insert the new message
            database.execute(
                "INSERT INTO messages (conversation_id, sender, content, created_at) VALUES (?, ?, ?, ?)",
                (conversation_id, sender, content, datetime.now().isoformat())
            )
            
            # Keep only the last max_messages for this conversation
            self._trim_conversation(conversation_id)
            
        except Exception as e:
            print(f"Warning: Failed to add message to buffer: {e}")
    
    def _trim_conversation(self, conversation_id: int) -> None:
        """Keep only the last N messages for a conversation."""
        try:
            # Get message count for this conversation
            count_result = database.query(
                "SELECT COUNT(*) as count FROM messages WHERE conversation_id = ?",
                (conversation_id,)
            )
            
            if count_result and count_result[0]["count"] > self.max_messages:
                # Delete oldest messages, keeping only the last max_messages
                database.execute(
                    """DELETE FROM messages WHERE conversation_id = ? AND id NOT IN (
                        SELECT id FROM messages WHERE conversation_id = ? 
                        ORDER BY created_at DESC LIMIT ?
                    )""",
                    (conversation_id, conversation_id, self.max_messages)
                )
        except Exception as e:
            print(f"Warning: Failed to trim conversation: {e}")
    
    def get_conversation_history(self, conversation_id: int) -> List[Dict[str, Any]]:
        """Get the conversation history for a conversation ID."""
        try:
            messages = database.query(
                "SELECT sender, content, created_at FROM messages WHERE conversation_id = ? "
                "ORDER BY created_at ASC",
                (conversation_id,)
            )
            return messages or []
        except Exception as e:
            print(f"Warning: Failed to get conversation history: {e}")
            return []
    
    def get_conversation_context(self, conversation_id: int) -> str:
        """Get formatted conversation context for LLM prompts."""
        messages = self.get_conversation_history(conversation_id)
        
        if not messages:
            return "No previous conversation context."
        
        context_lines = []
        for msg in messages:
            context_lines.append(f"{msg['sender']}: {msg['content']}")
        
        return "Previous conversation:\n" + "\n".join(context_lines)


class EntityMemory:
    """Manages entity-specific memory using the customer_kv table."""
    
    @staticmethod
    def set_customer_attribute(customer_id: int, key: str, value: str) -> None:
        """Set an attribute for a customer."""
        try:
            database.execute(
                "INSERT OR REPLACE INTO customer_kv (customer_id, key, value) VALUES (?, ?, ?)",
                (customer_id, key, value)
            )
        except Exception as e:
            print(f"Warning: Failed to set customer attribute: {e}")
    
    @staticmethod
    def get_customer_attribute(customer_id: int, key: str) -> Optional[str]:
        """Get an attribute for a customer."""
        try:
            result = database.query(
                "SELECT value FROM customer_kv WHERE customer_id = ? AND key = ?",
                (customer_id, key)
            )
            return result[0]["value"] if result else None
        except Exception as e:
            print(f"Warning: Failed to get customer attribute: {e}")
            return None
    
    @staticmethod
    def get_customer_profile(customer_id: int) -> Dict[str, str]:
        """Get all attributes for a customer."""
        try:
            results = database.query(
                "SELECT key, value FROM customer_kv WHERE customer_id = ?",
                (customer_id,)
            )
            return {result["key"]: result["value"] for result in results} if results else {}
        except Exception as e:
            print(f"Warning: Failed to get customer profile: {e}")
            return {}
    
    @staticmethod
    def update_last_contact(customer_id: int) -> None:
        """Update the last contact date for a customer."""
        EntityMemory.set_customer_attribute(
            customer_id, 
            "last_contact_date", 
            datetime.now().isoformat()
        )
    
    @staticmethod
    def set_customer_preference(customer_id: int, preference_type: str, value: str) -> None:
        """Set a customer preference."""
        EntityMemory.set_customer_attribute(customer_id, f"pref_{preference_type}", value)
    
    @staticmethod
    def get_customer_preference(customer_id: int, preference_type: str) -> Optional[str]:
        """Get a customer preference."""
        return EntityMemory.get_customer_attribute(customer_id, f"pref_{preference_type}")


class ConversationManager:
    """High-level conversation management."""
    
    def __init__(self):
        self.buffer = ConversationBuffer(max_messages=5)
    
    def start_conversation(self, user_id: int = 1) -> int:
        """Start a new conversation and return its ID."""
        try:
            cursor = database.execute(
                "INSERT INTO conversations (user_id, started_at) VALUES (?, ?)",
                (user_id, datetime.now().isoformat())
            )
            return cursor.lastrowid
        except Exception as e:
            print(f"Warning: Failed to start conversation: {e}")
            return 1  # Fallback to conversation ID 1
    
    def add_user_message(self, conversation_id: int, message: str) -> None:
        """Add a user message to the conversation."""
        self.buffer.add_message(conversation_id, "user", message)
    
    def add_agent_response(self, conversation_id: int, response: str, agent_name: str = "system") -> None:
        """Add an agent response to the conversation."""
        self.buffer.add_message(conversation_id, agent_name, response)
    
    def get_context_for_agent(self, conversation_id: int) -> str:
        """Get formatted context for agent processing."""
        return self.buffer.get_conversation_context(conversation_id)


# Global conversation manager instance
conversation_manager = ConversationManager()
