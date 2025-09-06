"""
Approval system for governance and compliance workflows.

This module handles approval workflows for high-value transactions
and other operations that require management approval.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Dict, Optional

import database
from tools.audit_logger import GlobalState


class ApprovalSystem:
    """Manages approval workflows for various operations."""
    
    # Approval thresholds (can be configured)
    INVOICE_APPROVAL_THRESHOLD = 1000.0  # $1000+
    PAYMENT_APPROVAL_THRESHOLD = 5000.0  # $5000+
    
    @staticmethod
    def requires_approval(operation_type: str, amount: float = 0.0, **kwargs) -> bool:
        """Check if an operation requires approval."""
        if operation_type == "invoice" and amount >= ApprovalSystem.INVOICE_APPROVAL_THRESHOLD:
            return True
        elif operation_type == "payment" and amount >= ApprovalSystem.PAYMENT_APPROVAL_THRESHOLD:
            return True
        
        # Add more approval rules as needed
        return False
    
    @staticmethod
    def request_approval(
        module: str,
        operation_type: str,
        payload: Dict[str, Any],
        requested_by: str = "system"
    ) -> int:
        """Request approval for an operation."""
        try:
            # Create approval request
            cursor = database.execute(
                "INSERT INTO approvals (module, payload_json, status, requested_by, created_at) "
                "VALUES (?, ?, 'pending', ?, ?)",
                (module, json.dumps(payload), requested_by, datetime.now().isoformat())
            )
            approval_id = cursor.lastrowid
            
            # Track in global state
            GlobalState.add_pending_approval(approval_id)
            
            return approval_id
            
        except Exception as e:
            print(f"Warning: Failed to request approval: {e}")
            return 0
    
    @staticmethod
    def get_approval_status(approval_id: int) -> Optional[str]:
        """Get the status of an approval request."""
        try:
            result = database.query(
                "SELECT status FROM approvals WHERE id = ?",
                (approval_id,)
            )
            return result[0]["status"] if result else None
        except Exception:
            return None
    
    @staticmethod
    def approve_request(approval_id: int, decided_by: str = "manager") -> bool:
        """Approve a pending request."""
        try:
            database.execute(
                "UPDATE approvals SET status = 'approved', decided_by = ?, decided_at = ? WHERE id = ?",
                (decided_by, datetime.now().isoformat(), approval_id)
            )
            return True
        except Exception as e:
            print(f"Warning: Failed to approve request: {e}")
            return False
    
    @staticmethod
    def reject_request(approval_id: int, decided_by: str = "manager") -> bool:
        """Reject a pending request."""
        try:
            database.execute(
                "UPDATE approvals SET status = 'rejected', decided_by = ?, decided_at = ? WHERE id = ?",
                (decided_by, datetime.now().isoformat(), approval_id)
            )
            return True
        except Exception as e:
            print(f"Warning: Failed to reject request: {e}")
            return False
    
    @staticmethod
    def get_pending_approvals() -> list[Dict[str, Any]]:
        """Get all pending approval requests."""
        try:
            results = database.query(
                "SELECT id, module, payload_json, requested_by, created_at "
                "FROM approvals WHERE status = 'pending' ORDER BY created_at ASC",
                None
            )
            
            approvals = []
            for result in results:
                approval = dict(result)
                # Parse the JSON payload
                try:
                    approval["payload"] = json.loads(approval["payload_json"])
                except:
                    approval["payload"] = {}
                del approval["payload_json"]
                approvals.append(approval)
            
            return approvals
        except Exception as e:
            print(f"Warning: Failed to get pending approvals: {e}")
            return []
    
    @staticmethod
    def format_approval_message(operation_type: str, amount: float, details: Dict[str, Any]) -> str:
        """Format a user-friendly approval message."""
        if operation_type == "invoice":
            return (
                f"This invoice creation for ${amount:.2f} requires management approval "
                f"(threshold: ${ApprovalSystem.INVOICE_APPROVAL_THRESHOLD:.2f}). "
                f"Your request has been submitted and is pending approval. "
                f"You will be notified once a decision is made."
            )
        elif operation_type == "payment":
            return (
                f"This payment of ${amount:.2f} requires management approval "
                f"(threshold: ${ApprovalSystem.PAYMENT_APPROVAL_THRESHOLD:.2f}). "
                f"Your request has been submitted and is pending approval."
            )
        else:
            return (
                f"This {operation_type} operation requires management approval. "
                f"Your request has been submitted and is pending review."
            )


def check_and_handle_approval(module: str, operation_type: str, amount: float = 0.0, **operation_data) -> tuple[bool, Optional[str]]:
    """
    Check if operation needs approval and handle accordingly.
    
    Returns:
        (needs_approval, approval_message)
        - If needs_approval is False, proceed with operation
        - If needs_approval is True, return the approval_message to user
    """
    if not ApprovalSystem.requires_approval(operation_type, amount):
        return False, None
    
    # Create approval request
    payload = {
        "operation_type": operation_type,
        "amount": amount,
        **operation_data
    }
    
    approval_id = ApprovalSystem.request_approval(module, operation_type, payload)
    
    if approval_id:
        message = ApprovalSystem.format_approval_message(operation_type, amount, operation_data)
        message += f" (Approval ID: {approval_id})"
        return True, message
    else:
        return True, "Failed to submit approval request. Please contact support."
