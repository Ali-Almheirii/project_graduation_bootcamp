"""
Audit logging system for tracking tool calls and system operations.

This module provides decorators and utilities for logging all tool calls
to the database for audit trails and compliance requirements.
"""

from __future__ import annotations

import json
import functools
from datetime import datetime
from typing import Any, Callable, Dict, Optional

import database


class AuditLogger:
    """Centralized audit logging system."""
    
    @staticmethod
    def log_tool_call(
        agent: str,
        tool_name: str,
        input_data: Dict[str, Any],
        output_data: Any,
        error: Optional[str] = None
    ) -> None:
        """Log a tool call to the database."""
        try:
            # Prepare data for storage
            input_json = json.dumps(input_data, default=str)
            
            if error:
                output_json = json.dumps({"error": error}, default=str)
            else:
                output_json = json.dumps({"result": output_data}, default=str)
            
            # Insert into tool_calls table
            database.execute(
                "INSERT INTO tool_calls (agent, tool_name, input_json, output_json, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (agent, tool_name, input_json, output_json, datetime.now().isoformat())
            )
            
        except Exception as e:
            # Don't let audit logging break the main functionality
            print(f"Warning: Failed to log tool call: {e}")


def audit_tool_call(agent_name: str, tool_name: str):
    """Decorator to automatically audit tool calls."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Capture input
            input_data = {
                "args": args,
                "kwargs": kwargs
            }
            
            try:
                # Execute the function
                result = func(*args, **kwargs)
                
                # Log successful call
                AuditLogger.log_tool_call(
                    agent=agent_name,
                    tool_name=tool_name,
                    input_data=input_data,
                    output_data=result
                )
                
                return result
                
            except Exception as e:
                # Log failed call
                AuditLogger.log_tool_call(
                    agent=agent_name,
                    tool_name=tool_name,
                    input_data=input_data,
                    output_data=None,
                    error=str(e)
                )
                
                # Re-raise the exception
                raise
                
        return wrapper
    return decorator


def audit_agent_method(agent_name: str):
    """Decorator to audit agent methods."""
    def decorator(func: Callable) -> Callable:
        method_name = func.__name__
        return audit_tool_call(agent_name, method_name)(func)
    return decorator


# Global state tracking
class GlobalState:
    """Track global system state."""
    
    @staticmethod
    def set_last_module(module_name: str) -> None:
        """Set the last used module globally."""
        try:
            # Use a simple key-value approach in customer_kv table for global state
            database.execute(
                "INSERT OR REPLACE INTO customer_kv (customer_id, key, value) VALUES (0, 'last_module', ?)",
                (module_name,)
            )
        except Exception as e:
            print(f"Warning: Failed to set last module: {e}")
    
    @staticmethod
    def get_last_module() -> Optional[str]:
        """Get the last used module."""
        try:
            result = database.query(
                "SELECT value FROM customer_kv WHERE customer_id = 0 AND key = 'last_module'",
                None
            )
            return result[0]["value"] if result else None
        except Exception:
            return None
    
    @staticmethod
    def add_pending_approval(approval_id: int) -> None:
        """Track a pending approval."""
        try:
            database.execute(
                "INSERT OR REPLACE INTO customer_kv (customer_id, key, value) VALUES (0, 'pending_approval', ?)",
                (str(approval_id),)
            )
        except Exception as e:
            print(f"Warning: Failed to track pending approval: {e}")
    
    @staticmethod
    def get_pending_approvals() -> list[int]:
        """Get all pending approval IDs."""
        try:
            results = database.query(
                "SELECT value FROM customer_kv WHERE customer_id = 0 AND key = 'pending_approval'",
                None
            )
            return [int(result["value"]) for result in results]
        except Exception:
            return []
