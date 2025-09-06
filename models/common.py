"""
Pydantic models shared across the API.

These models define the shape of requests and responses for the FastAPI
endpoints.  They provide input validation and generate OpenAPI schemas
automatically.
"""

from __future__ import annotations

from typing import Optional, List

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """Request body for the `/chat` endpoint."""

    message: str = Field(..., description="User message to the system")
    conversation_id: Optional[int] = Field(
        None,
        description=(
            "Conversation identifier.  If omitted, a new conversation is created."
        ),
    )


class ChatResponse(BaseModel):
    """Response body for the `/chat` endpoint."""

    conversation_id: int
    response: str


class ListOrdersResponse(BaseModel):
    orders: List[dict]


class CreateOrderRequest(BaseModel):
    customer_id: int
    items: List[dict]  # each item: {product_id, quantity}


class CreateOrderResponse(BaseModel):
    order_id: int
    message: str
