"""
Analytics & Reporting agent.

This agent answers quantitative and reasoning‑based questions by executing
SQL queries and summarising the results.  It supports listing saved
reports and running a specific saved report by its ID.  More complex
analytics can be added by extending the keyword heuristics or by
integrating a text‑to‑SQL model via the LLM.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from llm import call_gemini_prompt, LLMError
from tools.sql_tool import SQLTool
from tools.vector_rag_tool import VectorRAGTool, DefinitionRAGTool
from tools.analytics_tools import AnalyticsReportingTool, TextToSQLTool
from tools.saved_reports import SavedReportsManager
from tools.audit_logger import audit_agent_method


class AnalyticsAgent:
    def __init__(self) -> None:
        self.sql = SQLTool("analytics")
        self.rag = VectorRAGTool("main_documents")  # General RAG
        
        # Domain-specific RAG tools as per requirements
        try:
            self.definition_rag = DefinitionRAGTool()
        except Exception as e:
            print(f"Warning: Could not initialize DefinitionRAGTool: {e}")
            self.definition_rag = None

    def handle_query(self, query: str, conversation_id: Optional[int] = None) -> str:
        q = query.lower()
        try:
            if "list reports" in q or "saved reports" in q:
                return self._list_saved_reports(query)
            if ("revenue" in q or "profit" in q or "sales" in q) and ("day" in q or "daily" in q or "highest" in q or "best" in q):
                return self._analyze_revenue_by_day(query)
            if ("revenue" in q or "profit" in q or "sales" in q) and ("month" in q or "monthly" in q):
                return self._analyze_revenue_by_month(query)
            if ("revenue" in q or "profit" in q or "sales" in q) and ("total" in q or "overall" in q):
                return self._analyze_total_revenue(query)
            # Run report if the query contains "report" followed by an ID
            m = re.search(r"report\s+(\d+)", q)
            if m:
                report_id = int(m.group(1))
                return self._run_saved_report(report_id, query)
            # Fallback RAG search across glossary and documents
            rag_results = self.rag.search(query)
            if rag_results:
                return self._summarise(
                    f"RAG search results for '{query}': {rag_results}", query
                )
            return "I'm sorry, I couldn't understand your analytics request."
        except Exception:
            raise

    def _summarise(self, raw_response: str, query: str) -> str:
        prompt = (
            "You are an analytics assistant summarising data for a user.  "
            "Given the following raw information and the user's query, "
            "compose a concise and insightful reply.\n\n"
            f"User query: {query}\n"
            f"Raw information: {raw_response}"
        )
        return call_gemini_prompt(prompt)

    def _list_saved_reports(self, query: str) -> str:
        rows = self.sql.read(
            "SELECT id, title FROM saved_reports ORDER BY created_at DESC"
        )
        return self._summarise(str(rows), query)

    @audit_agent_method("analytics")
    def _create_advanced_analysis(self, query: str) -> str:
        """Create advanced analysis with chart generation."""
        try:
            # Use the analytics reporting tool to create a complete report
            report = AnalyticsReportingTool.create_report(query, domain="analytics")
            
            if report.get("chart") and report["chart"].get("type") != "message":
                chart_info = report["chart"]
                chart_desc = f"Generated {chart_info['type']} chart: {chart_info.get('title', 'Untitled')}"
                
                if chart_info.get("insights"):
                    insights = "\n".join([f"• {insight}" for insight in chart_info["insights"]])
                    chart_desc += f"\n\nKey insights:\n{insights}"
                
                return f"{report['narrative']}\n\n{chart_desc}\n\nChart data: {len(report['data'])} records analyzed."
            else:
                return report["narrative"]
                
        except Exception as e:
            return f"Failed to create advanced analysis: {str(e)}"
    
    @audit_agent_method("analytics") 
    def _list_available_reports(self) -> str:
        """List all available saved reports."""
        try:
            reports = SavedReportsManager.list_all_reports()
            
            if not reports:
                return "No saved reports are currently available."
            
            # Group by category
            categories = {}
            for report in reports:
                category = report["category"]
                if category not in categories:
                    categories[category] = []
                categories[category].append(report)
            
            response = "Available Saved Reports:\n\n"
            for category, category_reports in categories.items():
                response += f"**{category.title()} Reports:**\n"
                for report in category_reports:
                    response += f"  • Report {report['id']}: {report['name']} - {report['description']}\n"
                response += "\n"
            
            response += "To run a report, say 'run report [ID]' or 'show report [ID]'"
            return response
            
        except Exception as e:
            return f"Failed to list reports: {str(e)}"

    @audit_agent_method("analytics")
    def _run_saved_report(self, report_id: int, query: str) -> str:
        """Run a saved report by ID."""
        try:
            report_result = SavedReportsManager.execute_report(report_id)
            
            if report_result.get("error"):
                return report_result["error"]
            
            # Format the response
            response = f"Report: {report_result['report_name']}\n"
            response += f"Description: {report_result['description']}\n\n"
            response += report_result["narrative"]
            
            if report_result.get("chart") and report_result["chart"].get("type") != "message":
                chart_info = report_result["chart"]
                response += f"\n\nVisualization: {chart_info['type']} chart with {len(report_result['data'])} data points"
                
                if chart_info.get("insights"):
                    insights = "\n".join([f"• {insight}" for insight in chart_info["insights"]])
                    response += f"\n\nKey insights:\n{insights}"
            
            return response
            
        except Exception as e:
            return f"Failed to run report {report_id}: {str(e)}"

    def _analyze_revenue_by_day(self, query: str) -> str:
        """Analyze revenue by day to find highest revenue days."""
        rows = self.sql.read(
            "SELECT DATE(created_at) as date, SUM(total) as daily_revenue, COUNT(*) as order_count "
            "FROM orders WHERE status != 'cancelled' "
            "GROUP BY DATE(created_at) "
            "ORDER BY daily_revenue DESC LIMIT 5"
        )
        return self._summarise(str(rows), query)

    def _analyze_revenue_by_month(self, query: str) -> str:
        """Analyze revenue by month."""
        rows = self.sql.read(
            "SELECT strftime('%Y-%m', created_at) as month, SUM(total) as monthly_revenue, COUNT(*) as order_count "
            "FROM orders WHERE status != 'cancelled' "
            "GROUP BY strftime('%Y-%m', created_at) "
            "ORDER BY monthly_revenue DESC LIMIT 5"
        )
        return self._summarise(str(rows), query)

    def _analyze_total_revenue(self, query: str) -> str:
        """Analyze total revenue and key metrics."""
        rows = self.sql.read(
            "SELECT "
            "SUM(CASE WHEN status != 'cancelled' THEN total ELSE 0 END) as total_revenue, "
            "COUNT(CASE WHEN status != 'cancelled' THEN 1 END) as successful_orders, "
            "COUNT(CASE WHEN status = 'cancelled' THEN 1 END) as cancelled_orders, "
            "AVG(CASE WHEN status != 'cancelled' THEN total ELSE NULL END) as avg_order_value "
            "FROM orders"
        )
        return self._summarise(str(rows), query)
