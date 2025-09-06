"""
Streamlit front‑end for the agent‑driven ERP system.

This script provides a minimal chat interface.  It maintains a
conversation ID across messages and sends user input to the FastAPI
back‑end.  Responses are displayed in a simple chat log.
"""

import os
import json

import streamlit as st
import requests


API_URL = os.environ.get("ERP_API_URL", "http://localhost:8000")


def send_message(message: str, conversation_id: int | None) -> dict:
    """Send a chat message to the back‑end and return the JSON response."""
    payload = {"message": message, "conversation_id": conversation_id}
    resp = requests.post(f"{API_URL}/chat", json=payload)
    resp.raise_for_status()
    return resp.json()


def main() -> None:
    st.set_page_config(page_title="ERP Chat", page_icon="💬")
    st.title("Agent‑Driven ERP Chat")
    if "conversation_id" not in st.session_state:
        st.session_state.conversation_id = None
    if "history" not in st.session_state:
        st.session_state.history = []  # list of (sender, message)
    # Chat input
    user_input = st.chat_input("Ask me about sales, finance, inventory, or analytics…")
    if user_input:
        # Append user message to history
        st.session_state.history.append(("user", user_input))
        try:
            result = send_message(user_input, st.session_state.conversation_id)
            st.session_state.conversation_id = result["conversation_id"]
            st.session_state.history.append(("agent", result["response"]))
        except Exception as e:
            st.session_state.history.append(("agent", f"Error: {e}"))
    # Display history
    for sender, msg in st.session_state.history:
        if sender == "user":
            st.write(f"**You:** {msg}")
        else:
            st.write(f"**Agent:** {msg}")


if __name__ == "__main__":
    main()
