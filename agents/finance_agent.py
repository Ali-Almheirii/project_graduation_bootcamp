"""
Finance agent.

This agent automates financial workflows such as listing invoices and
payments and creating new invoices.  It delegates to SQL helpers for
persistence and uses the LLM to produce human‑friendly summaries of the
results.  Heuristic keyword matching is used to map user queries to
operations; you can extend or replace this with additional LLM calls for
classification if desired.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional

from llm import call_gemini_prompt, LLMError
from tools.sql_tool import SQLTool
from tools.vector_rag_tool import VectorRAGTool, PolicyRAGTool
from tools.approval_system import check_and_handle_approval
from tools.ml_tools import AnomalyDetectionTool
from tools.audit_logger import audit_agent_method
from tools.memory_manager import EntityMemory


class FinanceAgent:
    def __init__(self) -> None:
        self.sql = SQLTool("finance")
        self.rag = VectorRAGTool("main_documents")  # General RAG
        
        # Domain-specific RAG tools as per requirements
        try:
            self.policy_rag = PolicyRAGTool()
        except Exception as e:
            print(f"Warning: Could not initialize PolicyRAGTool: {e}")
            self.policy_rag = None

    def handle_query(self, query: str, conversation_id: Optional[int] = None) -> str:
        q = query.lower()
        try:
            if "invoices" in q and ("list" in q or "show" in q):
                return self._list_invoices(query)
            if "payments" in q and ("list" in q or "show" in q):
                return self._list_payments(query)
            if "invoice" in q and ("create" in q or "new" in q):
                return self._create_intelligent_invoice(query)
            # Check for policy questions first
            if any(keyword in q for keyword in ["policy", "refund", "procedure", "how", "what is", "what are"]):
                return self._handle_policy_query(query)
            # Fallback RAG search
            rag_results = self.rag.search(query)
            if rag_results:
                return self._summarise(
                    f"RAG search results for '{query}': {rag_results}", query
                )
            return "I'm sorry, I couldn't understand your finance request."
        except Exception:
            raise

    def _summarise(self, raw_response: str, query: str) -> str:
        prompt = (
            "You are a finance assistant summarising data for a user.  "
            "Given the following raw information and the user's query, "
            "compose a concise and professional reply.\n\n"
            f"User query: {query}\n"
            f"Raw information: {raw_response}"
        )
        return call_gemini_prompt(prompt)

    def _list_invoices(self, query: str) -> str:
        rows = self.sql.read(
            "SELECT invoices.id, customers.name AS customer_name, invoices.invoice_number, invoices.total_amount, invoices.status, invoices.issue_date "
            "FROM invoices JOIN customers ON invoices.customer_id = customers.id "
            "ORDER BY invoices.issue_date DESC LIMIT 5"
        )
        return self._summarise(str(rows), query)

    def _handle_policy_query(self, query: str) -> str:
        """Handle policy and procedure questions using RAG."""
        try:
            # Try domain-specific policy RAG first
            if self.policy_rag:
                rag_results = self.policy_rag.search(query, k=3)
                if rag_results:
                    return self._summarise_policy_results(rag_results, query)
            
            # Fallback to general RAG
            rag_results = self.rag.search(query, k=3, tags="policy")
            if rag_results:
                return self._summarise_policy_results(rag_results, query)
            
            return f"I couldn't find specific policy information about '{query}'. Please contact the finance department for clarification."
            
        except Exception as e:
            print(f"Warning: Policy RAG search failed: {e}")
            return f"I'm having trouble accessing policy information for '{query}'. Please try again or contact finance support."

    def _summarise_policy_results(self, results: list, query: str) -> str:
        """Summarize policy RAG results into a helpful response."""
        if not results:
            return f"I couldn't find policy information about '{query}'."
        
        # Build context from results
        context = ""
        for i, result in enumerate(results[:3], 1):
            context += f"{i}. {result.get('excerpt', '')}\n"
        
        # Use LLM to provide a structured response
        prompt = f"""
Based on the following finance policy information, answer the user's question: "{query}"

Relevant policy information:
{context}

Provide a clear, professional response that addresses the user's question about our finance policies.
"""
        
        try:
            response = call_gemini_prompt(prompt)
            return response.strip()
        except Exception as e:
            # Fallback to simple concatenation
            return f"Based on our finance policies:\n\n{context}"

    def _list_payments(self, query: str) -> str:
        rows = self.sql.read(
            "SELECT payments.id, customers.name AS customer_name, payments.amount, payments.method, payments.received_at "
            "FROM payments JOIN customers ON payments.customer_id = customers.id "
            "ORDER BY payments.received_at DESC LIMIT 5"
        )
        return self._summarise(str(rows), query)

    @audit_agent_method("finance")
    def _create_intelligent_invoice(self, query: str) -> str:
        """Create an invoice with amount extracted from the query."""
        import re
        
        # Extract amount from query using regex
        amount_match = re.search(r'\$?(\d+(?:\.\d{2})?)', query)
        if amount_match:
            requested_amount = float(amount_match.group(1))
        else:
            # Fallback to dummy creation if no amount found
            return self._create_dummy_invoice(query)
        
        try:
            # Pick a random customer
            customers = self.sql.read("SELECT id, name FROM customers")
            if not customers:
                raise RuntimeError("No customers available to create an invoice")
            customer = random.choice(customers)
            customer_id = customer["id"]
            customer_name = customer["name"]
            
            # Check if approval is needed
            needs_approval, approval_message = check_and_handle_approval(
                module="finance",
                operation_type="invoice",
                amount=requested_amount,
                customer_id=customer_id,
                customer_name=customer_name
            )
            
            if needs_approval:
                return approval_message
            
            invoice_number = f"INV{random.randint(100000,999999)}"
            
            # Create invoice with the requested amount
            invoice_id = self.sql.write(
                "INSERT INTO invoices (customer_id, invoice_number, issue_date, due_date, total_amount, status, created_at) "
                "VALUES (?, ?, date('now'), date('now','+30 day'), ?, 'unpaid', datetime('now'))",
                (customer_id, invoice_number, requested_amount),
            )
            
            # Update customer last contact
            EntityMemory.update_last_contact(customer_id)
            
            return f"Successfully created invoice {invoice_number} for {customer_name} with amount ${requested_amount:.2f}. Invoice ID: {invoice_id}"
            
        except Exception as e:
            return f"Failed to create invoice: {str(e)}"

    def _create_dummy_invoice(self, query: str) -> str:
        """Create a dummy invoice for demonstration purposes.

        The invoice is created for a random customer and associated with a
        random existing order.  It inserts a new invoice header and an entry
        into the `invoice_orders` link table.  Invoice lines are not created
        here.  Returns a summarised confirmation message.
        """
        # Pick a random customer
        customers = self.sql.read("SELECT id FROM customers")
        if not customers:
            raise RuntimeError("No customers available to create an invoice")
        customer_id = random.choice(customers)["id"]
        # Pick a random order for this customer if available
        orders = self.sql.read(
            "SELECT id, total FROM orders WHERE customer_id = ?", (customer_id,)
        )
        if not orders:
            raise RuntimeError("Selected customer has no orders to invoice")
        order = random.choice(orders)
        order_id = order["id"]
        total = order["total"]
        invoice_number = f"INV{random.randint(100000,999999)}"
        invoice_id = self.sql.write(
            "INSERT INTO invoices (customer_id, invoice_number, issue_date, due_date, total_amount, status, created_at) "
            "VALUES (?, ?, date('now'), date('now','+30 day'), ?, 'unpaid', datetime('now'))",
            (customer_id, invoice_number, total),
        )
        # Link invoice to order
        self.sql.write(
            "INSERT INTO invoice_orders (invoice_id, order_id) VALUES (?, ?)",
            (invoice_id, order_id),
        )
        raw_resp = {
            "invoice_id": invoice_id,
            "customer_id": customer_id,
            "order_id": order_id,
            "total": total,
            "invoice_number": invoice_number,
        }
        return self._summarise(f"New invoice created: {raw_resp}", query)
