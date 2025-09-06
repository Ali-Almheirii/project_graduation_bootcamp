"""
Saved reports system for predefined analytics and reporting.

This module manages predefined reports that can be executed by ID
and provides templates for common business analytics.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Dict, List, Optional

import database
from tools.analytics_tools import AnalyticsReportingTool, TextToSQLTool


class SavedReportsManager:
    """Manages saved reports and templates."""
    
    @staticmethod
    def initialize_default_reports() -> None:
        """Initialize the database with default predefined reports."""
        default_reports = [
            {
                "name": "Daily Revenue Summary",
                "description": "Total revenue and order count for today",
                "sql_template": "SELECT COUNT(*) as order_count, COALESCE(SUM(total), 0) as total_revenue FROM orders WHERE DATE(created_at) = DATE('now')",
                "chart_type": "bar",
                "category": "finance",
                "parameters": []
            },
            {
                "name": "Weekly Revenue Trend",
                "description": "Revenue trend over the last 7 days",
                "sql_template": "SELECT DATE(created_at) as date, COALESCE(SUM(total), 0) as daily_revenue FROM orders WHERE created_at >= DATE('now', '-7 days') GROUP BY DATE(created_at) ORDER BY date",
                "chart_type": "line",
                "category": "finance",
                "parameters": []
            },
            {
                "name": "Top Customers by Revenue",
                "description": "Customers ranked by total order value",
                "sql_template": "SELECT c.name, COALESCE(SUM(o.total), 0) as total_spent FROM customers c LEFT JOIN orders o ON c.id = o.customer_id GROUP BY c.id, c.name ORDER BY total_spent DESC LIMIT 10",
                "chart_type": "bar",
                "category": "sales",
                "parameters": []
            },
            {
                "name": "Lead Conversion Rate",
                "description": "Lead status distribution and conversion metrics",
                "sql_template": "SELECT status, COUNT(*) as count FROM leads GROUP BY status ORDER BY count DESC",
                "chart_type": "pie",
                "category": "sales",
                "parameters": []
            },
            {
                "name": "Monthly Invoice Status",
                "description": "Invoice status breakdown for current month",
                "sql_template": "SELECT status, COUNT(*) as count, COALESCE(SUM(total_amount), 0) as total_amount FROM invoices WHERE strftime('%Y-%m', issue_date) = strftime('%Y-%m', 'now') GROUP BY status",
                "chart_type": "pie",
                "category": "finance",
                "parameters": []
            },
            {
                "name": "Low Stock Alert",
                "description": "Products with low inventory levels",
                "sql_template": "SELECT p.name, s.quantity, s.location FROM products p JOIN stock s ON p.id = s.product_id WHERE s.quantity < 10 ORDER BY s.quantity ASC",
                "chart_type": "table",
                "category": "inventory",
                "parameters": []
            },
            {
                "name": "Recent Support Tickets",
                "description": "Support tickets from the last 30 days by status",
                "sql_template": "SELECT status, priority, COUNT(*) as count FROM tickets WHERE created_at >= DATE('now', '-30 days') GROUP BY status, priority ORDER BY count DESC",
                "chart_type": "bar",
                "category": "support",
                "parameters": []
            },
            {
                "name": "Customer Growth Trend",
                "description": "New customer registrations over time",
                "sql_template": "SELECT DATE(created_at) as date, COUNT(*) as new_customers FROM customers WHERE created_at >= DATE('now', '-30 days') GROUP BY DATE(created_at) ORDER BY date",
                "chart_type": "line",
                "category": "sales",
                "parameters": []
            }
        ]
        
        # Check if reports already exist
        existing = database.query("SELECT COUNT(*) as count FROM saved_reports", None)
        if existing and existing[0]["count"] > 0:
            print("Saved reports already initialized")
            return
        
        # Insert default reports (using existing schema: id, title, sql)
        for report in default_reports:
            try:
                # Use existing schema with title and sql columns
                database.execute(
                    "INSERT INTO saved_reports (title, sql) VALUES (?, ?)",
                    (report["name"], report["sql_template"])
                )
            except Exception as e:
                print(f"Warning: Failed to create report '{report['name']}': {e}")
        
        print("Default saved reports initialized")
    
    @staticmethod
    def get_report_by_id(report_id: int) -> Optional[Dict[str, Any]]:
        """Get a saved report by ID."""
        try:
            results = database.query(
                "SELECT id, title, sql FROM saved_reports WHERE id = ?",
                (report_id,)
            )
            
            if results:
                report = dict(results[0])
                # Add default values for missing columns
                report["name"] = report["title"]
                report["description"] = f"Saved report: {report['title']}"
                report["sql_template"] = report["sql"]
                report["chart_type"] = "table"  # Default chart type
                report["category"] = "general"
                report["parameters"] = []
                return report
            return None
            
        except Exception as e:
            print(f"Warning: Failed to get report {report_id}: {e}")
            return None
    
    @staticmethod
    def get_reports_by_category(category: str) -> List[Dict[str, Any]]:
        """Get all reports in a category."""
        try:
            results = database.query(
                "SELECT id, name, description, category FROM saved_reports WHERE category = ? ORDER BY name",
                (category,)
            )
            return [dict(result) for result in results] if results else []
            
        except Exception as e:
            print(f"Warning: Failed to get reports for category {category}: {e}")
            return []
    
    @staticmethod
    def list_all_reports() -> List[Dict[str, Any]]:
        """Get all saved reports with basic info."""
        try:
            results = database.query(
                "SELECT id, title FROM saved_reports ORDER BY title",
                None
            )
            
            reports = []
            for result in results:
                report = dict(result)
                report["name"] = report["title"]
                report["description"] = f"Saved report: {report['title']}"
                report["category"] = "general"
                report["chart_type"] = "table"
                reports.append(report)
            
            return reports
            
        except Exception as e:
            print(f"Warning: Failed to list reports: {e}")
            return []
    
    @staticmethod
    def execute_report(report_id: int, parameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute a saved report and return results with visualization."""
        report = SavedReportsManager.get_report_by_id(report_id)
        
        if not report:
            return {
                "error": f"Report {report_id} not found",
                "data": [],
                "chart": None
            }
        
        try:
            # Execute the SQL template
            sql_query = report["sql_template"]
            
            # TODO: Parameter substitution could be added here if needed
            # For now, we'll execute the template as-is
            
            data = database.query(sql_query, None)
            
            if not data:
                return {
                    "report_id": report_id,
                    "report_name": report["name"],
                    "data": [],
                    "chart": {
                        "type": "message",
                        "title": report["name"],
                        "message": "No data available for this report"
                    },
                    "narrative": "No data found for this report at this time."
                }
            
            # Generate chart specification
            chart_spec = AnalyticsReportingTool.generate_chart_spec(
                data,
                chart_type=report["chart_type"],
                title=report["name"],
                query_context=report["description"]
            )
            
            # Generate narrative
            narrative = f"Report '{report['name']}' executed successfully. {report['description']}. Found {len(data)} records."
            
            return {
                "report_id": report_id,
                "report_name": report["name"],
                "description": report["description"],
                "category": report["category"],
                "data": data,
                "chart": chart_spec,
                "narrative": narrative,
                "executed_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"Warning: Failed to execute report {report_id}: {e}")
            return {
                "error": f"Failed to execute report: {str(e)}",
                "report_id": report_id,
                "report_name": report.get("name", "Unknown"),
                "data": [],
                "chart": None
            }
    
    @staticmethod
    def search_reports(query: str) -> List[Dict[str, Any]]:
        """Search reports by name or description."""
        try:
            results = database.query(
                "SELECT id, name, description, category FROM saved_reports "
                "WHERE name LIKE ? OR description LIKE ? ORDER BY name",
                (f"%{query}%", f"%{query}%")
            )
            return [dict(result) for result in results] if results else []
            
        except Exception as e:
            print(f"Warning: Failed to search reports: {e}")
            return []


# Convenience functions
def run_report(report_id: int) -> Dict[str, Any]:
    """Quick report execution."""
    return SavedReportsManager.execute_report(report_id)


def list_reports() -> List[Dict[str, Any]]:
    """Quick report listing."""
    return SavedReportsManager.list_all_reports()


def find_reports(category: str) -> List[Dict[str, Any]]:
    """Quick category search."""
    return SavedReportsManager.get_reports_by_category(category)


# Initialize reports on module import
try:
    SavedReportsManager.initialize_default_reports()
except Exception as e:
    print(f"Warning: Could not initialize saved reports: {e}")
