"""
Analytics and reporting tools for business intelligence.

This module provides text-to-SQL conversion and chart generation
capabilities for the analytics agent.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

from llm import call_gemini_prompt
import database


class TextToSQLTool:
    """Converts natural language queries to SQL."""
    
    @staticmethod
    def generate_sql(
        query: str,
        domain: str = "general",
        schema_context: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate SQL from natural language query."""
        
        # Default schema context
        if not schema_context:
            schema_context = """
Available tables and key columns:
- customers: id, name, email, phone, created_at
- orders: id, customer_id, total, status, created_at
- products: id, name, price, description
- invoices: id, customer_id, total_amount, status, issue_date
- payments: id, customer_id, amount, method, received_at
- leads: id, customer_name, contact_email, score, status, created_at
- stock: product_id, quantity, location
"""
        
        prompt = f"""
You are an expert SQL query generator. Convert this natural language query to SQL:

Query: "{query}"
Domain: {domain}

Database Schema:
{schema_context}

Requirements:
- Generate SQLite-compatible SQL
- Use appropriate JOINs when needed
- Include relevant columns for the query
- Use proper aggregation functions
- Add appropriate ORDER BY and LIMIT clauses
- Ensure the query is safe (no DROP, DELETE without WHERE, etc.)

Respond with JSON:
{{
    "sql": "SELECT ...",
    "explanation": "This query retrieves...",
    "confidence": "high",
    "tables_used": ["customers", "orders"],
    "potential_issues": ["None" or list of issues]
}}

Confidence levels: low, medium, high
"""
        
        try:
            response = call_gemini_prompt(prompt)
            result = json.loads(response.strip())
            
            # Validate SQL is present
            if not result.get("sql"):
                raise ValueError("No SQL generated")
            
            return result
            
        except Exception as e:
            print(f"Warning: SQL generation failed: {e}")
            return {
                "sql": "SELECT 'Error: Could not generate SQL' as message",
                "explanation": f"Failed to generate SQL: {str(e)}",
                "confidence": "low",
                "tables_used": [],
                "potential_issues": ["SQL generation failed"]
            }
    
    @staticmethod
    def execute_generated_sql(query: str, domain: str = "general") -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Generate and execute SQL from natural language."""
        sql_result = TextToSQLTool.generate_sql(query, domain)
        
        try:
            # Execute the SQL
            sql_query = sql_result["sql"]
            data = database.query(sql_query, None)
            return data or [], sql_result
            
        except Exception as e:
            print(f"Warning: SQL execution failed: {e}")
            return [], {
                **sql_result,
                "execution_error": str(e),
                "confidence": "low"
            }


class AnalyticsReportingTool:
    """Generates chart specifications and formatted reports."""
    
    @staticmethod
    def generate_chart_spec(
        data: List[Dict[str, Any]],
        chart_type: str = "auto",
        title: str = "",
        query_context: str = ""
    ) -> Dict[str, Any]:
        """Generate chart specification for data visualization."""
        
        if not data:
            return {
                "type": "message",
                "title": title or "No Data",
                "message": "No data available for visualization",
                "data": []
            }
        
        # Analyze data structure
        sample_row = data[0]
        columns = list(sample_row.keys())
        
        # Prepare data summary for LLM
        data_summary = f"Data has {len(data)} rows with columns: {', '.join(columns)}\n"
        data_summary += f"Sample row: {sample_row}\n"
        
        if len(data) > 1:
            data_summary += f"Last row: {data[-1]}\n"
        
        prompt = f"""
You are a data visualization expert. Analyze this data and create a chart specification:

Query Context: {query_context}
{data_summary}

Chart Type Preference: {chart_type}

Create a chart specification that best represents this data. Consider:
- Data types (numerical, categorical, dates)
- Number of data points
- Relationships in the data
- Best visualization type for the query context

Respond with JSON:
{{
    "type": "bar|line|pie|table|scatter",
    "title": "Chart Title",
    "x_axis": "column_name",
    "y_axis": "column_name", 
    "x_label": "X Axis Label",
    "y_label": "Y Axis Label",
    "data": [prepared data for chart],
    "summary": "Brief description of what the chart shows",
    "insights": ["Key insight 1", "Key insight 2"]
}}

Chart Types:
- bar: categorical data, comparisons
- line: time series, trends
- pie: proportions, percentages  
- table: detailed data, multiple columns
- scatter: correlations, relationships
"""
        
        try:
            response = call_gemini_prompt(prompt)
            result = json.loads(response.strip())
            
            # Ensure we have the original data
            result["raw_data"] = data
            result["row_count"] = len(data)
            
            # If no title provided, use the generated one
            if not title and result.get("title"):
                result["title"] = result["title"]
            elif title:
                result["title"] = title
            
            return result
            
        except Exception as e:
            print(f"Warning: Chart generation failed: {e}")
            # Fallback to table format
            return {
                "type": "table",
                "title": title or "Data Table",
                "data": data,
                "raw_data": data,
                "row_count": len(data),
                "summary": f"Table view of {len(data)} records",
                "insights": ["Data visualization failed, showing raw table"]
            }
    
    @staticmethod
    def create_report(
        query: str,
        domain: str = "general",
        include_chart: bool = True
    ) -> Dict[str, Any]:
        """Create a complete analytics report with data and visualization."""
        
        # Generate and execute SQL
        data, sql_info = TextToSQLTool.execute_generated_sql(query, domain)
        
        # Create chart if requested and data is available
        chart_spec = None
        if include_chart and data:
            chart_spec = AnalyticsReportingTool.generate_chart_spec(
                data, 
                title=f"Analysis: {query}",
                query_context=query
            )
        
        # Generate narrative summary
        narrative = AnalyticsReportingTool._generate_narrative(query, data, sql_info)
        
        return {
            "query": query,
            "sql_info": sql_info,
            "data": data,
            "chart": chart_spec,
            "narrative": narrative,
            "timestamp": database.query("SELECT datetime('now') as now", None)[0]["now"]
        }
    
    @staticmethod
    def _generate_narrative(query: str, data: List[Dict[str, Any]], sql_info: Dict[str, Any]) -> str:
        """Generate a narrative summary of the analysis."""
        
        if not data:
            return f"No data found for the query: '{query}'. This could indicate that there are no records matching the criteria or the query needs to be refined."
        
        data_summary = f"Found {len(data)} records. "
        
        # Add basic insights based on data
        if len(data) == 1:
            data_summary += "Here are the details for the single result found."
        elif len(data) < 10:
            data_summary += "Here is a summary of the key findings."
        else:
            data_summary += "Here are the key insights from the analysis."
        
        # Add SQL confidence info
        confidence = sql_info.get("confidence", "medium")
        if confidence == "high":
            data_summary += " The analysis was performed with high confidence in the data accuracy."
        elif confidence == "low":
            data_summary += " Please note that this analysis has lower confidence due to query complexity or data limitations."
        
        return data_summary


# Convenience functions
def text_to_sql(query: str, domain: str = "general") -> str:
    """Quick text-to-SQL conversion returning just the SQL."""
    result = TextToSQLTool.generate_sql(query, domain)
    return result.get("sql", "")


def quick_analysis(query: str, domain: str = "general") -> Dict[str, Any]:
    """Quick analysis with chart generation."""
    return AnalyticsReportingTool.create_report(query, domain, include_chart=True)


def table_only_analysis(query: str, domain: str = "general") -> List[Dict[str, Any]]:
    """Quick analysis returning just the data table."""
    data, _ = TextToSQLTool.execute_generated_sql(query, domain)
    return data
