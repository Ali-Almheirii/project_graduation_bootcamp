"""
LLM-based ML tools for predictive analytics and scoring.

This module provides ML-like functionality using LLM reasoning
instead of traditional machine learning models for simplicity.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from llm import call_gemini_prompt


class LeadScoringTool:
    """LLM-based lead scoring tool."""
    
    @staticmethod
    def score_lead(
        customer_name: str,
        contact_email: str,
        message: str = "",
        company_info: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Score a lead using LLM analysis."""
        
        # Prepare context for LLM
        context = f"""
Lead Information:
- Customer Name: {customer_name}
- Email: {contact_email}
- Message/Inquiry: {message or "No message provided"}
"""
        
        if company_info:
            context += f"\nCompany Information:\n"
            for key, value in company_info.items():
                context += f"- {key}: {value}\n"
        
        prompt = f"""
You are a lead scoring expert. Analyze the following lead and provide a score from 0.0 to 1.0 based on these criteria:

{context}

Scoring Criteria:
- Email quality (professional domain vs generic): 0.0-0.2
- Company name/type (established vs individual): 0.0-0.2  
- Message quality (specific inquiry vs generic): 0.0-0.3
- Contact information completeness: 0.0-0.1
- Overall business potential: 0.0-0.2

Provide your analysis in JSON format:
{{
    "score": 0.75,
    "confidence": "high",
    "reasoning": "Professional email domain, established company name, specific inquiry about services",
    "risk_factors": ["None identified"],
    "next_actions": ["Schedule follow-up call", "Send product information"]
}}
"""
        
        try:
            response = call_gemini_prompt(prompt)
            # Try to parse JSON response
            result = json.loads(response.strip())
            
            # Ensure score is within bounds
            score = max(0.0, min(1.0, result.get("score", 0.5)))
            result["score"] = score
            
            return result
            
        except Exception as e:
            print(f"Warning: Lead scoring failed: {e}")
            # Fallback scoring
            return {
                "score": 0.5,
                "confidence": "low",
                "reasoning": "Unable to analyze lead automatically",
                "risk_factors": ["Analysis failed"],
                "next_actions": ["Manual review required"]
            }


class AnomalyDetectionTool:
    """LLM-based anomaly detection for financial transactions."""
    
    @staticmethod
    def detect_anomalies(
        transaction_data: List[Dict[str, Any]],
        context: str = "financial_transaction"
    ) -> Dict[str, Any]:
        """Detect anomalies in transaction patterns using LLM analysis."""
        
        if not transaction_data:
            return {"anomalies": [], "risk_level": "low", "summary": "No data to analyze"}
        
        # Prepare transaction summary for LLM
        transactions_text = "Recent Transactions:\n"
        for i, tx in enumerate(transaction_data[:10], 1):  # Limit to 10 most recent
            amount = tx.get("amount", tx.get("total", 0))
            date = tx.get("created_at", tx.get("date", "Unknown"))
            customer = tx.get("customer_name", tx.get("customer", "Unknown"))
            transactions_text += f"{i}. ${amount:.2f} - {customer} - {date}\n"
        
        prompt = f"""
You are a financial fraud detection expert. Analyze these transaction patterns for anomalies:

{transactions_text}

Look for:
- Unusual amounts (much higher/lower than typical)
- Suspicious timing patterns
- Duplicate or near-duplicate transactions
- Round number amounts that seem artificial
- Customer behavior anomalies

Provide analysis in JSON format:
{{
    "anomalies": [
        {{
            "transaction_index": 1,
            "type": "unusual_amount",
            "severity": "medium",
            "description": "Amount significantly higher than typical"
        }}
    ],
    "risk_level": "medium",
    "summary": "2 potential anomalies detected requiring review",
    "recommendations": ["Review high-value transactions", "Verify customer identity"]
}}

Risk levels: low, medium, high
Anomaly types: unusual_amount, suspicious_timing, duplicate_transaction, round_amount, customer_anomaly
"""
        
        try:
            response = call_gemini_prompt(prompt)
            result = json.loads(response.strip())
            return result
            
        except Exception as e:
            print(f"Warning: Anomaly detection failed: {e}")
            return {
                "anomalies": [],
                "risk_level": "unknown",
                "summary": "Analysis failed",
                "recommendations": ["Manual review required"]
            }


class ForecastingTool:
    """LLM-based demand forecasting tool."""
    
    @staticmethod
    def forecast_demand(
        product_id: int,
        historical_data: List[Dict[str, Any]],
        forecast_period: int = 30
    ) -> Dict[str, Any]:
        """Forecast demand for a product using LLM analysis."""
        
        if not historical_data:
            return {
                "forecast": 0,
                "confidence": "low",
                "trend": "unknown",
                "reasoning": "No historical data available"
            }
        
        # Prepare historical data for LLM
        history_text = f"Historical demand data for Product ID {product_id}:\n"
        for i, record in enumerate(historical_data[-12:], 1):  # Last 12 records
            quantity = record.get("quantity", record.get("demand", 0))
            date = record.get("date", record.get("created_at", f"Period {i}"))
            history_text += f"{i}. {quantity} units - {date}\n"
        
        prompt = f"""
You are a demand forecasting expert. Analyze this historical demand data and predict future demand:

{history_text}

Forecast period: {forecast_period} days

Consider:
- Seasonal patterns
- Growth/decline trends  
- Recent changes in demand
- Overall stability of demand

Provide forecast in JSON format:
{{
    "forecast": 150,
    "confidence": "medium",
    "trend": "increasing",
    "seasonal_factor": 1.1,
    "reasoning": "Steady upward trend with slight seasonal increase",
    "risk_factors": ["Supply chain disruptions"],
    "recommendations": ["Increase stock by 20%", "Monitor supplier capacity"]
}}

Confidence levels: low, medium, high
Trends: increasing, decreasing, stable, volatile
"""
        
        try:
            response = call_gemini_prompt(prompt)
            result = json.loads(response.strip())
            
            # Ensure forecast is non-negative
            forecast = max(0, result.get("forecast", 0))
            result["forecast"] = forecast
            
            return result
            
        except Exception as e:
            print(f"Warning: Demand forecasting failed: {e}")
            # Simple fallback: average of recent demand
            try:
                recent_quantities = [
                    record.get("quantity", record.get("demand", 0)) 
                    for record in historical_data[-5:]
                ]
                avg_demand = sum(recent_quantities) / len(recent_quantities) if recent_quantities else 0
                
                return {
                    "forecast": int(avg_demand),
                    "confidence": "low",
                    "trend": "stable",
                    "reasoning": "Fallback to recent average due to analysis failure"
                }
            except:
                return {
                    "forecast": 0,
                    "confidence": "low",
                    "trend": "unknown",
                    "reasoning": "Unable to generate forecast"
                }


# Convenience functions for easy integration
def score_lead(customer_name: str, contact_email: str, message: str = "") -> float:
    """Quick lead scoring function returning just the score."""
    result = LeadScoringTool.score_lead(customer_name, contact_email, message)
    return result.get("score", 0.5)


def detect_transaction_anomalies(transactions: List[Dict[str, Any]]) -> List[str]:
    """Quick anomaly detection returning list of anomaly descriptions."""
    result = AnomalyDetectionTool.detect_anomalies(transactions)
    anomalies = result.get("anomalies", [])
    return [anomaly.get("description", "Unknown anomaly") for anomaly in anomalies]


def forecast_product_demand(product_id: int, history: List[Dict[str, Any]]) -> int:
    """Quick demand forecasting returning forecasted quantity."""
    result = ForecastingTool.forecast_demand(product_id, history)
    return result.get("forecast", 0)
