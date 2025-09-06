"""
Inventory & Supply Chain agent.

This agent manages stock levels, suppliers and purchase orders.  It
interprets user queries with simple keyword heuristics and calls the
database to retrieve or modify data.  After performing an operation it
invokes the LLM to summarise the result for the user.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional

from llm import call_gemini_prompt, LLMError
from tools.sql_tool import SQLTool
from tools.vector_rag_tool import VectorRAGTool, DocRAGTool


class InventoryAgent:
    def __init__(self) -> None:
        self.sql = SQLTool("inventory")
        self.rag = VectorRAGTool("main_documents")  # General RAG
        
        # Domain-specific RAG tools as per requirements
        try:
            self.doc_rag = DocRAGTool()
        except Exception as e:
            print(f"Warning: Could not initialize DocRAGTool: {e}")
            self.doc_rag = None

    def handle_query(self, query: str, conversation_id: Optional[int] = None) -> str:
        q = query.lower()
        try:
            if ("stock" in q or "inventory" in q) and ("list" in q or "show" in q or "levels" in q or "status" in q):
                return self._list_stock(query)
            if ("product" in q or "products" in q) and ("expensive" in q or "cheapest" in q or "price" in q or "pricing" in q):
                return self._list_products_by_price(query)
            if "suppliers" in q and ("list" in q or "show" in q):
                return self._list_suppliers(query)
            if ("purchase order" in q or "po" in q) and ("create" in q or "new" in q):
                return self._create_dummy_po(query)
            # Fallback RAG search
            rag_results = self.rag.search(query)
            if rag_results:
                return self._summarise(
                    f"RAG search results for '{query}': {rag_results}", query
                )
            return "I'm sorry, I couldn't understand your inventory request."
        except Exception:
            raise

    def _summarise(self, raw_response: str, query: str) -> str:
        prompt = (
            "You are an inventory assistant summarising data for a user.  "
            "Given the following raw information and the user's query, "
            "compose a concise and helpful reply.\n\n"
            f"User query: {query}\n"
            f"Raw information: {raw_response}"
        )
        return call_gemini_prompt(prompt)

    def _list_stock(self, query: str) -> str:
        rows = self.sql.read(
            "SELECT products.name AS product_name, stock.qty_on_hand, stock.reorder_point "
            "FROM stock JOIN products ON stock.product_id = products.id "
            "ORDER BY stock.qty_on_hand ASC LIMIT 5"
        )
        return self._summarise(str(rows), query)

    def _list_products_by_price(self, query: str) -> str:
        q = query.lower()
        if "expensive" in q or "highest" in q:
            # Show most expensive products
            rows = self.sql.read(
                "SELECT products.name, products.price, products.description, stock.qty_on_hand "
                "FROM products LEFT JOIN stock ON products.id = stock.product_id "
                "ORDER BY products.price DESC LIMIT 5"
            )
        elif "cheapest" in q or "lowest" in q:
            # Show cheapest products
            rows = self.sql.read(
                "SELECT products.name, products.price, products.description, stock.qty_on_hand "
                "FROM products LEFT JOIN stock ON products.id = stock.product_id "
                "ORDER BY products.price ASC LIMIT 5"
            )
        else:
            # Show products with pricing info
            rows = self.sql.read(
                "SELECT products.name, products.price, products.description, stock.qty_on_hand "
                "FROM products LEFT JOIN stock ON products.id = stock.product_id "
                "ORDER BY products.price DESC LIMIT 5"
            )
        return self._summarise(str(rows), query)

    def _list_suppliers(self, query: str) -> str:
        rows = self.sql.read(
            "SELECT id, name, email, phone FROM suppliers ORDER BY name LIMIT 5"
        )
        return self._summarise(str(rows), query)

    def _create_dummy_po(self, query: str) -> str:
        """Create a dummy purchase order for demonstration.

        Select a random supplier and product and create a purchase order with
        a single line item.  Returns a summarised confirmation message.
        """
        suppliers = self.sql.read("SELECT id FROM suppliers")
        if not suppliers:
            raise RuntimeError("No suppliers available to create a purchase order")
        supplier_id = random.choice(suppliers)["id"]
        # Pick a random product
        products = self.sql.read("SELECT id, price FROM products")
        if not products:
            raise RuntimeError("No products available to create a purchase order")
        product = random.choice(products)
        product_id = product["id"]
        unit_cost = product["price"]  # using selling price as stand‑in for cost
        quantity = random.randint(1, 10)
        # Create PO header
        po_id = self.sql.write(
            "INSERT INTO purchase_orders (supplier_id, status, created_at) VALUES (?, 'draft', datetime('now'))",
            (supplier_id,),
        )
        # Create PO line
        self.sql.write(
            "INSERT INTO po_items (po_id, product_id, quantity, unit_cost) VALUES (?, ?, ?, ?)",
            (po_id, product_id, quantity, unit_cost),
        )
        raw_resp = {
            "po_id": po_id,
            "supplier_id": supplier_id,
            "product_id": product_id,
            "quantity": quantity,
            "unit_cost": unit_cost,
        }
        return self._summarise(f"New purchase order created: {raw_resp}", query)
