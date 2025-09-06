# Agent‑Driven ERP System

This repository contains a reference implementation of a modular, agent‑driven ERP platform built as part of a bootcamp graduation project.  The goal of the system is to demonstrate how large language models (LLMs) and composable tools can be combined to create a flexible, conversational interface over classic ERP functionality.  It is **not** a production‑ready system, but a well‑organised starting point for experimentation and future development.

## High‑Level Architecture

```
┌────────────┐    HTTP    ┌─────────────┐
│ Streamlit  │ ─────────▶ │  FastAPI     │
│   Front‑end│            │  Back‑end    │
└────────────┘            └─────┬───────┘
                                  │
             ┌────────────────────┼────────────────────┐
             │                    │                    │
        ┌────▼─────┐       ┌──────▼─────┐       ┌──────▼────┐
        │ Router   │       │ Domain     │ …     │ Domain    │
        │ Agent    │       │ Agent (X)  │       │ Agent (Y) │
        └──────────┘       └────────────┘       └───────────┘
             │
             ▼
         ┌───────────┐
         │ Tools &   │
         │ Utilities │
         └────┬──────┘
              │
         ┌────▼──────┐
         │  SQLite    │
         │  Database  │
         └────────────┘
```

1. **Streamlit Front‑end:** A light‑weight chat UI that sends user messages to the FastAPI server and displays responses.  It is intentionally minimal and can be extended or replaced by a more sophisticated React front‑end.

2. **FastAPI Back‑end (`app.py`):** Exposes a simple `/chat` endpoint for conversational input and a handful of domain‑specific endpoints (e.g. listing orders or invoices).  The back‑end instantiates the router and domain agents and manages shared resources like the database connection and LLM client.

3. **Router Agent (`agents/router_agent.py`):** Receives each user utterance, calls the LLM to determine which domain agent should handle the request and then delegates the request to that agent.  It also records tool invocations and approvals in the database.  The router is LLM‑driven and will raise an error if no `GEMINI_API_KEY` is provided or if the call fails.

4. **Domain Agents (`agents/*.py`):** Each domain agent encapsulates the logic for a particular business area:
   - **SalesAgent:** Manages customers, leads, orders and tickets.
   - **FinanceAgent:** Handles invoices, payments and ledger entries.
   - **InventoryAgent:** Maintains stock levels, purchase orders and supplier data.
   - **AnalyticsAgent:** Executes saved reports and ad‑hoc analytical queries.

5. **Tools (`tools/*.py`):** Provide reusable capabilities such as SQL read/write helpers (`sql_tool.py`) and a vector-based retrieval‑augmented generator (`vector_rag_tool.py`). The RAG tool uses sentence-transformers for embeddings and ChromaDB for vector storage, providing semantic search capabilities with domain-specific tools for Sales, Finance, Inventory, and Analytics.

6. **Database (`database.py`):** Centralised helper for opening and reusing a SQLite connection.  This project ships with `erp.db`, which contains a small set of pre‑populated tables described in the accompanying database documentation.  See the `project_data` directory for the sample database.

7. **LLM Integration (`llm.py`):** Contains a thin wrapper around Google’s Gemini API.  The wrapper reads your API key from the `GEMINI_API_KEY` environment variable.  If the key is missing or the API returns an error the call will raise an exception—no fallback logic is provided.  You can replace the implementation with a call to a local LLM via LM Studio by editing this file.

## Running Locally

1. Install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Set your Gemini API key:

```bash
export GEMINI_API_KEY=<your‑api‑key>
```

3. Start the FastAPI server:

```bash
uvicorn erp_app.app:app --reload
```

4. In a separate terminal run the Streamlit UI (if you have `streamlit` installed):

```bash
streamlit run erp_app/streamlit_app.py
```

5. Open the Streamlit interface in your browser (usually at `http://localhost:8501`) and start chatting.  Alternatively you can POST directly to the `/chat` endpoint using curl or Postman.

## Notes

* **Local vs. Cloud LLMs:** This implementation uses a cloud‑hosted Gemini endpoint by default.  If you prefer to run a local LLM (e.g. via LM Studio) you can modify `llm.py` to call your local server instead.  The rest of the system does not need to change.
* **Approvals and Logging:** The router logs tool calls and approvals into the `tool_calls` and `approvals` tables.  For brevity this reference implementation does not enforce approval flows; it only records them.
* **MCP Registry:** The original project spec references a Modular Composable Protocol (MCP) registry tool.  In this simplified implementation we provide a minimal registry inside the router that lists the available tools on each domain agent.  You can extend this to a full MCP server in the future.
* **Extensibility:** The code is organised for clarity over brevity.  Each agent exposes discrete methods that can be individually tested and reused.  Adding a new domain agent typically involves creating a new class in `agents/`, registering it in the router, and adding any necessary SQL helpers in `tools/`.
