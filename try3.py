# scalemancer.py
from langgraph.graph import StateGraph, END
from langchain.output_parsers import JsonOutputParser
from langchain.schema import OutputParserException
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
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
            "fetch_cpu", "fetch_users",
            "forecast_cpu", "forecast_users",
            "optimization_suggestion", "cost_analysis",
        ]
    ]
    start_time: Optional[datetime]
    end_time: Optional[datetime]
    server_role: Optional[str]
    cpu_df: Optional[object]
    user_df: Optional[object]
    cpu_forecast: Optional[object]
    user_forecast: Optional[object]
    avg_cpu: Optional[float]
    avg_users: Optional[float]
    cost: Optional[float]
    summary_md: Optional[str]
    suggestion: Optional[Literal["up", "down", "same"]]
    justification_md: Optional[str]
    confirmed: Optional[bool]
    result: Optional[str]

# ---------- LLM ----------
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=os.getenv("OPENAI_API_KEY"))
# llm = ChatOllama(model="llama3:8b", temperature=0)

# ---------- Dummy Functions ----------
def fetch_cpu_usage_data(start_time, end_time, server_role):
    print(f"Fetching CPU usage data for {server_role} from {start_time} to {end_time}")
    dates = pd.date_range(start=start_time, end=end_time, freq="D")
    values = [random.uniform(30, 90) for _ in dates]
    return pd.DataFrame({"timestamp": dates, "cpu_usage": values})

def fetch_active_customers_data(start_time, end_time, server_role):
    print(f"Fetching active users data for {server_role} from {start_time} to {end_time}")
    dates = pd.date_range(start=start_time, end=end_time, freq="D")
    values = [random.randint(500, 2000) for _ in dates]
    return pd.DataFrame({"timestamp": dates, "active_customers": values})

def fetch_azure_vm_specs() -> pd.DataFrame:
    vm_specs = [
        {"name": "Standard_D2s_v3", "cpu_cores": 2, "memory_gb": 8, "cost_per_month_usd": 70},
        {"name": "Standard_D4s_v3", "cpu_cores": 4, "memory_gb": 16, "cost_per_month_usd": 140},
        {"name": "Standard_D8s_v3", "cpu_cores": 8, "memory_gb": 32, "cost_per_month_usd": 280},
    ]
    return pd.DataFrame(vm_specs)

def forecast_with_prophet(df, column_name):
    df = df.rename(columns={"timestamp": "ds", column_name: "y"})
    model = Prophet()
    model.fit(df)
    future = model.make_future_dataframe(periods=7)
    forecast = model.predict(future)
    return forecast[["ds", "yhat"]]

def trigger_scaling_action(suggestion):
    time.sleep(2)
    return f"Scaling action '{suggestion}' successfully triggered."

# ---------- LangGraph Nodes ----------
def intent_detection_node(state):
    print("Detecting intent using LLM...")

    query = state["query"]

    prompt = f"""
You are an infrastructure system analyst assistant.

Identify:
- Intent: one of ["fetch_cpu", "fetch_users", "forecast_cpu", "forecast_users", "optimization_suggestion", "cost_analysis"]
- Start time and end time (YYYY-MM-DD format) if available
- Server role (e.g., "web server", "database") if mentioned

ONLY return JSON like:
{{
  "intent": "forecast_cpu",
  "start_time": "2024-04-01",
  "end_time": "2024-04-15",
  "server_role": "web server"
}}

If not mentioned, set fields to null.

Query: ```{query}```
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
        return {**state, **parsed}

    except Exception as e:
        print("Error parsing intent detection:", e)
        raise

def fetch_data_node(state):
    print("Fetching historical data...")
    start_time = state.get("start_time") or datetime.today() - timedelta(days=90)
    end_time = state.get("end_time") or datetime.today()
    server_role = state.get("server_role") or "all"

    cpu_df = fetch_cpu_usage_data(start_time, end_time, server_role)
    user_df = fetch_active_customers_data(start_time, end_time, server_role)
    azure_vm_df = fetch_azure_vm_specs(start_time, end_time, server_role)

    return {**state, "cpu_df": cpu_df, "user_df": user_df, "azure_vm_df": azure_vm_df}

def forecast_node(state):
    print("Forecasting future trends...")
    cpu_forecast = forecast_with_prophet(state["cpu_df"], "cpu_usage")
    user_forecast = forecast_with_prophet(state["user_df"], "active_customers")
    return {**state, "cpu_forecast": cpu_forecast, "user_forecast": user_forecast}

def analysis_node(state):
    print("Analyzing trends and estimating cost...")
    avg_cpu = state.get("cpu_forecast", pd.DataFrame()).get("yhat", pd.Series([0])).mean()
    avg_users = state.get("user_forecast", pd.DataFrame()).get("yhat", pd.Series([0])).mean()
    cost = avg_cpu * 0.05 + avg_users * 0.01

    md = f"""### Analysis Summary
- Avg CPU Forecast: {avg_cpu:.2f}%
- Avg Active Users Forecast: {avg_users:.0f}
- Estimated Cost: ${cost:.2f}
"""
    return {**state, "avg_cpu": avg_cpu, "avg_users": avg_users, "cost": cost, "summary_md": md}

def decision_node(state):
    print("Making scaling decision...")

    prompt = f"""
Based on:
- Avg CPU: {state.get('avg_cpu')}
- Avg Users: {state.get('avg_users')}
- Estimated Cost: {state.get('cost')}

Decide: "up", "down", or "same" regarding scaling servers.
Give justification in markdown.

Respond ONLY as JSON:
{{
  "suggestion": "up",
  "justification": "CPU load expected to increase significantly."
}}
"""
    raw = llm.invoke(prompt).content.strip()
    parsed = json.loads(raw)
    return {**state, "suggestion": parsed["suggestion"], "justification_md": parsed["justification"]}

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

graph.add_node("Intent", RunnableLambda(intent_detection_node))
graph.add_node("FetchData", RunnableLambda(fetch_data_node))
graph.add_node("Forecast", RunnableLambda(forecast_node))
graph.add_node("Analyze", RunnableLambda(analysis_node))
graph.add_node("Decide", RunnableLambda(decision_node))
graph.add_node("Confirm", RunnableLambda(human_in_loop_node))
graph.add_node("Act", RunnableLambda(action_node))

graph.set_entry_point("Intent")

graph.add_conditional_edges("Intent", lambda s: s["intent"], {
    "fetch_cpu": "FetchData",
    "fetch_users": "FetchData",
    "forecast_cpu": "FetchData",
    "forecast_users": "FetchData",
    "optimization_suggestion": "FetchData",
    "cost_analysis": "FetchData",
})

graph.add_conditional_edges("FetchData", lambda s: s["intent"], {
    "fetch_cpu": "Analyze",
    "fetch_users": "Analyze",
    "forecast_cpu": "Forecast",
    "forecast_users": "Forecast",
    "optimization_suggestion": "Forecast",
    "cost_analysis": "Forecast",
})

graph.add_edge("Forecast", "Analyze")

graph.add_conditional_edges("Analyze", lambda s: s["intent"] in ["optimization_suggestion", "cost_analysis"], {
    True: "Decide",
    False: END
})

graph.add_conditional_edges("Decide", lambda s: s["suggestion"] != "same", {
    True: "Confirm",
    False: END
})

graph.add_conditional_edges("Confirm", lambda s: s["confirmed"], {
    True: "Act",
    False: END
})

graph.set_finish_point("Act")

app = graph.compile()

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
