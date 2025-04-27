# scalemancer.py
from langgraph.graph import StateGraph, END
from langchain_core.output_parsers.json import JsonOutputParser
from langchain.schema import OutputParserException
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from IPython.display import Image, display
from langchain_core.runnables.graph import CurveStyle, MermaidDrawMethod, NodeStyles
import pandas as pd
from prophet import Prophet
import random
import time
from typing import TypedDict, Literal, Optional
from datetime import datetime, timedelta
import os
import json
import re

# ---------- State Schema ----------
class ScaleMancerState(TypedDict, total=False):
    query: str
    intent: Optional[
        Literal[
            "analyze_trend", "forecast_future",
            "optimize_infra", "capacity_planning",
            "server_info", "unknown"
        ]
    ]
    start_time: Optional[datetime]
    end_time: Optional[datetime]
    server_role: Optional[str]
    forecast_period: Optional[int]  # Days to forecast
    cpu_df: Optional[object]
    user_df: Optional[object]
    ram_df: Optional[object]
    cpu_forecast: Optional[object]
    user_forecast: Optional[object]
    ram_forecast: Optional[object]
    avg_cpu: Optional[float]
    avg_users: Optional[float]
    avg_ram: Optional[float]
    cost: Optional[float]
    summary_md: Optional[str]
    suggestion: Optional[Literal["up", "down", "same"]]
    justification_md: Optional[str]
    insights: Optional[list]
    confirmed: Optional[bool]
    result: Optional[str]
    server_specs: Optional[object]

# ---------- LLM ----------
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=os.getenv("OPENAI_API_KEY"))
# llm = ChatOllama(model="llama3:8b", temperature=0)

# ---------- Dummy Functions ----------
def fetch_cpu_usage_data(start_time, end_time, server_role):
    print(f"Fetching CPU usage data for {server_role} from {start_time} to {end_time}")
    dates = pd.date_range(start=start_time, end=end_time, freq="D")
    values = [random.uniform(30, 90) for _ in dates]
    return pd.DataFrame({"timestamp": dates, "cpu_usage": values})

def fetch_ram_usage_data(start_time, end_time, server_role):
    print(f"Fetching RAM usage data for {server_role} from {start_time} to {end_time}")
    dates = pd.date_range(start=start_time, end=end_time, freq="D")
    values = [random.uniform(40, 85) for _ in dates]
    return pd.DataFrame({"timestamp": dates, "ram_usage": values})

def fetch_active_customers_data(start_time, end_time, server_role):
    print(f"Fetching active users data for {server_role} from {start_time} to {end_time}")
    dates = pd.date_range(start=start_time, end=end_time, freq="D")
    values = [random.randint(500, 2000) for _ in dates]
    return pd.DataFrame({"timestamp": dates, "active_customers": values})


def fetch_server_specs(server_role=None):
    print(f"Fetching server specifications for {server_role}")
    vm_specs = [
        {"name": "Standard_D2s_v3", "cpu_cores": 2, "memory_gb": 8, "cost_per_month_usd": 70},
        {"name": "Standard_D4s_v3", "cpu_cores": 4, "memory_gb": 16, "cost_per_month_usd": 140},
        {"name": "Standard_D8s_v3", "cpu_cores": 8, "memory_gb": 32, "cost_per_month_usd": 280},
    ]
    return pd.DataFrame(vm_specs)

def forecast_with_prophet(df, column_name, periods=7):
    df = df.rename(columns={"timestamp": "ds", column_name: "y"})
    model = Prophet()
    model.fit(df)
    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)
    return forecast[["ds", "yhat"]]

def trigger_scaling_action(suggestion):
    time.sleep(2)
    return f"Scaling action '{suggestion}' successfully triggered."

def detect_anomalies(df, column_name, threshold=2.0):
    series = df[column_name]
    mean = series.mean()
    std = series.std()
    anomalies = df[abs(series - mean) > threshold * std]
    return anomalies

# ---------- LangGraph Nodes ----------
def intent_detection_node(state):
    print("Detecting intent using LLM...")
    
    query = state["query"]
    
    prompt = f""" 
    You are an infrastructure system analyst assistant.

    Identify the user's intent from these categories:
    - analyze_trend: trend analysis, outage/anomaly detection
    - forecast_future: forecast usage, predict future performance
    - optimize_infra: identify cost-saving opportunities (downscaling, orphan VMs)
    - capacity_planning: recommend infra scaling for growth
    - server_info: server specs queries
    - unknown: fallback for unclear requests

    Also identify:
    - Start time and end time (YYYY-MM-DD format) if available
    - Server role (e.g., "web server", "database") if mentioned
    - Forecast period (number of days to forecast) if applicable

    ONLY return JSON like:
    {{
      "intent": "analyze_trend",
      "start_time": "2024-04-01",
      "end_time": "2024-04-15",
      "server_role": "web server",
      "forecast_period": 30
    }}

    If not mentioned, set fields to null.

    Query: 
    {query}

    Respond ONLY with JSON. No extra text.
    """
    
    raw_response = llm.invoke(prompt).content.strip()
    
    try:
        match = re.search(r"\{.*\}", raw_response, re.DOTALL)
        json_text = match.group(0) if match else raw_response
        parsed = json.loads(json_text)
        
        # Normalize dates
        def parse_date(d):
            return datetime.strptime(d, "%Y-%m-%d") if d else None
        
        parsed["start_time"] = parse_date(parsed.get("start_time"))
        parsed["end_time"] = parse_date(parsed.get("end_time"))
        
        # Set default forecast period if not provided
        if "forecast_period" not in parsed or not parsed["forecast_period"]:
            parsed["forecast_period"] = 7
            
        return {**state, **parsed}
    
    except Exception as e:
        print("Error parsing intent detection:", e)
        return {**state, "intent": "unknown"}
    
def analysis_node(state):
    print("Analyzing data using LLM...")
    
    query = state.get("query")
    intent = state.get("intent")
    cpu_df = state.get("cpu_df")
    user_df = state.get("user_df")
    ram_df = state.get("ram_df")
    cpu_forecast = state.get("cpu_forecast")  # Use .get()
    user_forecast = state.get("user_forecast") # Use .get()
    ram_forecast = state.get("ram_forecast")   # Use .get()
    server_role = state.get("server_role")
    
    prompt = f""" 
    You are an Analysis Expert responsible for understanding user queries related to server infrastructure.
    Based on the query's intent, you must:
    Analyze historical CPU usage, active user data, and server costs.
    Detect outages, anomalies, or performance degradation if the user asked about trend analysis.
    Forecast future CPU usage, active users, or server performance if the user asked for predictions.
    Estimate infrastructure cost based on CPU usage, server uptime, or user load if required.
    Summarize findings clearly in markdown format for reports.

    Always explain:
    What data was analyzed.
    Any important insights (outages, spikes, drops, trends, anomalies).
    Forecast trends, if forecasting is involved.
    Cost estimation if requested or applicable.
    Suggest scaling actions if needed.

    Inputs provided to you:
    cpu_df: CPU usage dataframe
    ram_df: RAM usage dataframe
    user_df: Active customers dataframe
    cpu_forecast: CPU usage forecast dataframe
    ram_forecast: RAM usage forecast dataframe
    user_forecast: Active customers forecast dataframe
    start_time and end_time
    intent: (one of: analyze_trend, forecast_future, optimize_infra, capacity_planning)
    server_role: (e.g., "web server", "database server", "app server", "all")

    query (user query in natural language)

    Important Rules:
    If intent is analyze_trend: Focus on past trends, outages, anomaly detection.
    If intent is forecast_future: Forecast CPU usage and active users into the future.
    If intent is optimize_infra: Focus on identifying low usage periods, suggest downscaling if applicable.
    If intent is capacity_planning: Suggest infra scaling up/down based on forecasted needs.

    Output Format:
    Return a brief markdown summary report analyzing this.
    Always return JSON with the following format:

    {{
      "summary_md": "### Analysis Summary\n- Points here...",
      "suggestion": "up"/"down"/"same",
      "justification_md": "### Justification\n- Points here...",
      "insights": ["CPU peaked on Sep 20", "Anomaly detected on Aug 15"],
      "estimated_cost": 150.00
    }}

    Input Data:
    cpu_df: {cpu_df}
    ram_df: {ram_df}
    user_df: {user_df}
    cpu_forecast: {cpu_forecast}
    ram_forecast: {ram_forecast}
    user_forecast: {user_forecast}
    intent: {intent}
    server_role: {server_role}

    Query: 
    {query}

    Respond ONLY with JSON. No extra text.
    """
    
    raw_response = llm.invoke(prompt).content.strip()
    
    try:
        match = re.search(r"\{.*\}", raw_response, re.DOTALL)
        json_text = match.group(0) if match else raw_response
        parsed = json.loads(json_text)
        
        # # Normalize dates
        # def parse_date(d):
        #     return datetime.strptime(d, "%Y-%m-%d") if d else None
        
        # parsed["start_time"] = parse_date(parsed.get("start_time"))
        # parsed["end_time"] = parse_date(parsed.get("end_time"))
        
        # # Set default forecast period if not provided
        # if "forecast_period" not in parsed or not parsed["forecast_period"]:
        #     parsed["forecast_period"] = 7
            
        return {**state, **parsed}
    
    except Exception as e:
        print("Error parsing intent detection:", e)
        return {**state, "intent": "unknown"}

def fetch_data_node(state):
    print("Fetching historical data...")
    start_time = state.get("start_time") or datetime.today() - timedelta(days=90)
    end_time = state.get("end_time") or datetime.today()
    server_role = state.get("server_role") or "all"
    
    cpu_df = fetch_cpu_usage_data(start_time, end_time, server_role)
    ram_df = fetch_ram_usage_data(start_time, end_time, server_role)
    user_df = fetch_active_customers_data(start_time, end_time, server_role)
    server_specs = fetch_server_specs(server_role)
    
    return {**state, "cpu_df": cpu_df, "user_df": user_df, "ram_df": ram_df, "server_specs": server_specs}

# def fetch_server_info_node(state):
#     print("Fetching server specifications...")
#     server_role = state.get("server_role") or "all"
#     server_specs = fetch_server_specs(server_role)

#     return {**state, "server_specs": server_specs}

def forecast_node(state):
    print("Forecasting future trends...")
    forecast_period = state.get("forecast_period", 7)
    
    cpu_forecast = forecast_with_prophet(state["cpu_df"], "cpu_usage", forecast_period)
    user_forecast = forecast_with_prophet(state["user_df"], "active_customers", forecast_period)
    ram_forecast = forecast_with_prophet(state["ram_df"], "ram_usage", forecast_period)
    
    return {**state, 
            "cpu_forecast": cpu_forecast, 
            "user_forecast": user_forecast,
            "ram_forecast": ram_forecast}

def trend_analysis_node(state):
    print("Analyzing historical trends and anomalies...")
    
    # Detect anomalies
    cpu_anomalies = detect_anomalies(state["cpu_df"], "cpu_usage")
    user_anomalies = detect_anomalies(state["user_df"], "active_customers")
    ram_anomalies = detect_anomalies(state["ram_df"], "ram_usage")
    
    # Calculate averages
    avg_cpu = state["cpu_df"]["cpu_usage"].mean()
    avg_users = state["user_df"]["active_customers"].mean()
    avg_ram = state["ram_df"]["ram_usage"].mean()
    
    anomaly_summary = ""
    if not cpu_anomalies.empty:
        anomaly_summary += f"\n- Found {len(cpu_anomalies)} CPU usage anomalies"
    if not user_anomalies.empty:
        anomaly_summary += f"\n- Found {len(user_anomalies)} user activity anomalies"
    if not ram_anomalies.empty:
        anomaly_summary += f"\n- Found {len(ram_anomalies)} RAM usage anomalies"
    
    md = f"""### Trend Analysis Summary
      - Average CPU Usage: {avg_cpu:.2f}%
      - Average Active Users: {avg_users:.0f}
      - Average RAM Usage: {avg_ram:.2f}%
      {anomaly_summary}
      """
    
    return {**state, 
            "avg_cpu": avg_cpu, 
            "avg_users": avg_users, 
            "avg_ram": avg_ram,
            "summary_md": md}

def cost_analysis_node(state):
    print("Analyzing costs...")
    
    # Calculate base costs
    avg_cpu = state.get("avg_cpu", 0)
    avg_users = state.get("avg_users", 0)
    avg_ram = state.get("avg_ram", 0)
    
    # Basic cost model
    cpu_cost_factor = 0.05
    user_cost_factor = 0.01
    ram_cost_factor = 0.03
    
    # Calculate forecasted costs
    if "cpu_forecast" in state:
        forecasted_cpu = state["cpu_forecast"]["yhat"].mean()
        forecasted_users = state["user_forecast"]["yhat"].mean()
        forecasted_ram = state["ram_forecast"]["yhat"].mean()
        
        current_cost = avg_cpu * cpu_cost_factor + avg_users * user_cost_factor + avg_ram * ram_cost_factor
        forecasted_cost = forecasted_cpu * cpu_cost_factor + forecasted_users * user_cost_factor + forecasted_ram * ram_cost_factor
        
        # Get current server specs
        server_specs = state.get("server_specs")
        if server_specs is not None:
            current_server_type = server_specs.iloc[0]["name"]
            current_server_cost = server_specs.iloc[0]["cost_per_month_usd"]
        else:
            current_server_type = "Unknown"
            current_server_cost = 0
        
        md = f"""### Cost Analysis Summary
- Current Server: {current_server_type} (${current_server_cost}/month)
- Current Resource Cost: ${current_cost:.2f}/day
- Forecasted Resource Cost: ${forecasted_cost:.2f}/day
- Cost Change: {((forecasted_cost - current_cost) / current_cost * 100):.2f}%
"""
    else:
        # Use only current data if no forecast is available
        cost = avg_cpu * cpu_cost_factor + avg_users * user_cost_factor + avg_ram * ram_cost_factor
        md = f"""### Cost Analysis Summary
- Current Resource Cost: ${cost:.2f}/day
"""
        forecasted_cost = cost
    
    return {**state, "cost": forecasted_cost, "summary_md": md}

def scaling_recommendation_node(state):
    print("Generating scaling recommendations...")
    
    # Extract relevant metrics
    forecasted_cpu = state["cpu_forecast"]["yhat"].mean() if "cpu_forecast" in state else state.get("avg_cpu", 0)
    forecasted_ram = state["ram_forecast"]["yhat"].mean() if "ram_forecast" in state else state.get("avg_ram", 0)
    forecasted_users = state["user_forecast"]["yhat"].mean() if "user_forecast" in state else state.get("avg_users", 0)
    cost = state.get("cost", 0)
    
    prompt = f""" 
As an infrastructure analyst, make a scaling recommendation based on:
- Forecasted CPU Usage: {forecasted_cpu:.2f}%
- Forecasted RAM Usage: {forecasted_ram:.2f}%
- Forecasted Active Users: {forecasted_users:.0f}
- Estimated Daily Cost: ${cost:.2f}

Decide: "up", "down", or "same" regarding scaling servers.
Provide detailed justification in markdown format.

Respond ONLY as JSON:
{{
  "suggestion": "up",
  "justification": "CPU load expected to increase significantly."
}}
"""
    
    raw = llm.invoke(prompt).content.strip()
    
    try:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        json_text = match.group(0) if match else raw
        parsed = json.loads(json_text)
        
        # Create a more detailed markdown summary
        md = f"""### Scaling Recommendation
- Recommendation: {parsed["suggestion"].upper()}
- Justification: {parsed["justification"]}

#### Metrics
- Forecasted CPU: {forecasted_cpu:.2f}%
- Forecasted RAM: {forecasted_ram:.2f}%
- Forecasted Users: {forecasted_users:.0f}
- Estimated Cost: ${cost:.2f}/day
"""
        
        return {**state, 
                "suggestion": parsed["suggestion"], 
                "justification_md": parsed["justification"],
                "summary_md": md}
    except Exception as e:
        print("Error parsing scaling recommendation:", e)
        return {**state, 
                "suggestion": "same", 
                "justification_md": "Error generating recommendation. Defaulting to no change.",
                "summary_md": state.get("summary_md", "")}

def unknown_intent_node(state):
    result = "Sorry, I couldn't understand. Could you rephrase your request?"
    return {**state, "result": result, "summary_md": result}

def human_in_loop_node(state):
    confirmation = input(f"Suggested action is '{state['suggestion']}'. Confirm? (yes/no): ")
    return {**state, "confirmed": confirmation.lower() == "yes"}

def action_node(state):
    if state.get("confirmed"):
        result = trigger_scaling_action(state["suggestion"])
        return {**state, "result": result}
    return {**state, "result": "Action not confirmed."}

# ---------- Graph Definition ----------
graph = StateGraph(ScaleMancerState)

# Add nodes to the graph
graph.add_node("Intent", RunnableLambda(intent_detection_node))
graph.add_node("FetchData", RunnableLambda(fetch_data_node))
graph.add_node("FetchServerInfo", RunnableLambda(fetch_data_node))
graph.add_node("Forecast", RunnableLambda(forecast_node))
graph.add_node("TrendAnalysis", RunnableLambda(trend_analysis_node))
graph.add_node("CostAnalysis", RunnableLambda(cost_analysis_node))
graph.add_node("Analyze", RunnableLambda(analysis_node))
graph.add_node("ScalingRecommendation", RunnableLambda(scaling_recommendation_node))
graph.add_node("Confirm", RunnableLambda(human_in_loop_node))
graph.add_node("Act", RunnableLambda(action_node))
graph.add_node("Unknown", RunnableLambda(unknown_intent_node))

# Set entry point
graph.set_entry_point("Intent")

# Define conditional edges based on intent
graph.add_conditional_edges("Intent", lambda s: s["intent"], {
    "analyze_trend": "FetchData",
    "forecast_future": "FetchData",
    "optimize_infra": "FetchData",
    "capacity_planning": "FetchData",
    "server_info": "FetchData",
    "unknown": END
})

# Define flow for analyze_trend intent
graph.add_conditional_edges("FetchData", lambda s: s["intent"], {
    "analyze_trend": "Analyze",
    "forecast_future": "Forecast",
    "optimize_infra": "Forecast",
    "capacity_planning": "Forecast",
})

# Define flow for forecast_future intent
graph.add_edge("Forecast", "Analyze")

# Define flow for optimize_infra intent
graph.add_conditional_edges("Analyze", lambda s: s["intent"], {
    "analyze_trend": END,
    "forecast_future": END,
    "optimize_infra": "ScalingRecommendation",
    "capacity_planning": "ScalingRecommendation",
})

# Define flow for optimize_infra intent
graph.add_conditional_edges("TrendAnalysis", lambda s: s["intent"], {
    "analyze_trend": END,
    "forecast_future": END,
    "optimize_infra": "FetchServerInfo",
    "capacity_planning": "FetchServerInfo",
})

graph.add_conditional_edges("FetchServerInfo", lambda s: s["intent"], {
    "server_info": END,
    "optimize_infra": "CostAnalysis",
    "capacity_planning": "CostAnalysis",
})

# Define flow for capacity_planning intent
graph.add_conditional_edges("CostAnalysis", lambda s: s["intent"], {
    "optimize_infra": END,
    "capacity_planning": "ScalingRecommendation",
})

graph.add_conditional_edges("ScalingRecommendation", lambda s: s["suggestion"] != "same", {
    True: "Confirm",
    False: END
})

graph.add_conditional_edges("Confirm", lambda s: s["confirmed"], {
    True: "Act",
    False: END
})

# Set finish points
graph.set_finish_point("Act")
graph.set_finish_point("Unknown")

# Compile the graph
app = graph.compile()
print(app.get_graph().draw_mermaid())

# ---------- Main ----------
if __name__ == "__main__":
    user_input = input("Enter your query: ")
    initial_state = {"query": user_input}
    final_state = app.invoke(initial_state)
    
    print("\n--- Markdown Report ---")
    print(final_state.get("summary_md", ""))
    
    if final_state.get("justification_md"):
        print("\n--- Justification ---")
        print(final_state["justification_md"])
    
    print("\n--- Final Result ---")
    print(final_state.get("result", "No result"))
    
    import pprint
    pprint.pprint(final_state, width=120)