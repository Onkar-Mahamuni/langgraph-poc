from langgraph.graph import StateGraph, END
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
    intent: Optional[str]
    start_time: Optional[str]
    end_time: Optional[str]
    server_role: Optional[str]
    cpu_df: Optional[object]
    user_df: Optional[object]
    cpu_forecast: Optional[object]
    user_forecast: Optional[object]
    avg_cpu: Optional[float]
    avg_users: Optional[float]
    cost: Optional[float]
    analysis_text: Optional[str]
    summary_md: Optional[str]
    suggestion: Optional[Literal["up", "down", "same"]]
    justification_md: Optional[str]
    confirmed: Optional[bool]
    result: Optional[str]

# llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=os.getenv("OPENAI_API_KEY"))
llm = ChatOllama(model="llama3.1:8b", temperature=0)

# ---------- Dummy Functions ----------
def fetch_cpu_usage_data(start_time, end_time, server_role):
    print(f"Fetching CPU usage data for {server_role} from {start_time} to {end_time}")
    dates = pd.date_range(start=start_time, end=end_time, freq="D")
    values = [random.uniform(30, 90) for _ in dates]
    return pd.DataFrame({"timestamp": dates, "cpu_usage": values})

def fetch_active_customers_data(start_time, end_time, server_role):
    print(f"Fetching active customers data for {server_role} from {start_time} to {end_time}")
    dates = pd.date_range(start=start_time, end=end_time, freq="D")
    values = [random.randint(500, 2000) for _ in dates]
    return pd.DataFrame({"timestamp": dates, "active_customers": values})

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
You are a system analyst assistant. A user has asked a question related to infrastructure usage and scaling.

Your tasks:
1. Classify the intent of the query as:
   - simple: querying past metrics (e.g., outages, performance)
   - normal: requesting trend analysis or cost insight
   - complex: involving forecasting, optimization, or hypothetical situations

2. Extract the time window (start and end date) if mentioned. If not, leave it empty.

3. Extract the server or server role if mentioned (e.g., "web servers", "database", etc.).

4. Return your response in the following JSON format:

{{
  "intent": "simple" | "normal" | "complex",
  "start_time": "YYYY-MM-DD" | null,
  "end_time": "YYYY-MM-DD" | null,
  "server": "<server role>" | null
}}

Query: ```{query}```
Return only the JSON object with no other explanation.
"""

    raw_response = llm.invoke(prompt).content.strip()

    # Try to extract JSON safely even if LLM wraps it in triple backticks
    try:
        match = re.search(r"\{.*\}", raw_response, re.DOTALL)
        json_text = match.group(0) if match else raw_response
        parsed = json.loads(json_text)

        # Normalize dates
        def parse_date(d):
            return datetime.strptime(d, "%Y-%m-%d") if d else None

        parsed["start_time"] = parse_date(parsed.get("start_time"))
        parsed["end_time"] = parse_date(parsed.get("end_time"))
        parsed["server_role"] = parsed.pop("server", None)  # 👈 Add this line
        return {**state, **parsed}

    except Exception as e:
        print("Error parsing intent response:", e)
        raise
  # Convert JSON string to dict

def fetch_data_node(state):
    print("Fetching data for forecasting...")

    end_time = state.get("end_time")
    start_time = state.get("start_time")

    if not end_time or pd.isna(end_time):
        end_time = datetime.today()
    if not start_time or pd.isna(start_time):
        start_time = end_time - timedelta(days=90)

    # Convert string dates to datetime
    if isinstance(start_time, str):
        start_time = pd.to_datetime(start_time)
    if isinstance(end_time, str):
        end_time = pd.to_datetime(end_time)

    # Default or provided server role
    server_role = state.get("server") or "all"

    print(f"Fetching data from {start_time.date()} to {end_time.date()} for server role: {server_role}")

    # Now correctly pass `server_role` to the functions
    cpu_df = fetch_cpu_usage_data(start_time, end_time, server_role)
    user_df = fetch_active_customers_data(start_time, end_time, server_role)

    return {**state, "start_time": start_time, "end_time": end_time, "cpu_df": cpu_df, "user_df": user_df}

def forecast_node(state):
    cpu_forecast = forecast_with_prophet(state["cpu_df"], "cpu_usage")
    user_forecast = forecast_with_prophet(state["user_df"], "active_customers")
    return {**state, "cpu_forecast": cpu_forecast, "user_forecast": user_forecast}

def analysis_node(state):
    md = f"### Analysis Report for {state['server_role']} ({state['start_time']} to {state['end_time']})\n"
    md += "- Data and Trends:\n"
    if state.get("cpu_forecast") is not None:
        avg_cpu = state["cpu_forecast"]["yhat"].mean()
        avg_users = state["user_forecast"]["yhat"].mean()
        cost_est = avg_cpu * 0.05 + avg_users * 0.01
        md += f"  - Avg CPU Forecast: {avg_cpu:.2f}%\n"
        md += f"  - Avg Users Forecast: {avg_users:.0f}\n"
        md += f"  - Estimated Cost: ${cost_est:.2f}\n"
        state.update({"avg_cpu": avg_cpu, "avg_users": avg_users, "cost": cost_est})
    else:
        md += "  - No forecasting performed (simple query).\n"
    return {**state, "summary_md": md}

def decision_node(state):
    prompt = f"""
    Based on:
    - CPU forecast avg: {state.get('avg_cpu')}
    - User forecast avg: {state.get('avg_users')}
    - Cost: {state.get('cost')}

    Should we scale up, down, or keep same? Justify in markdown.
    Respond in JSON: {{ "suggestion": "...", "justification": "..." }}
    """
    response = eval(llm.invoke(prompt).content)
    return {**state, "suggestion": response["suggestion"], "justification_md": response["justification"]}

def human_in_loop_node(state):
    confirmation = input(f"Suggested action is '{state['suggestion']}'. Confirm? (yes/no): ")
    return {**state, "confirmed": confirmation.lower() == "yes"}

def action_node(state):
    if state["confirmed"]:
        result = trigger_scaling_action(state["suggestion"])
        return {**state, "result": result}
    return {**state, "result": "Action not confirmed."}

# ---------- LangGraph Definition ----------
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
    "simple": "FetchData",
    "normal": "FetchData",
    "complex": "FetchData"
})
graph.add_conditional_edges("FetchData", lambda s: s["intent"], {
    "simple": "Analyze",
    "normal": "Forecast",
    "complex": "Forecast"
})
graph.add_edge("Forecast", "Analyze")
graph.add_conditional_edges("Analyze", lambda s: s.get("intent") != "simple", {
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

# ---------- Entry ----------
if __name__ == "__main__":
    user_input = input("Enter your query: ")
    state = {"query": user_input}
    final_state = app.invoke(state)
    print("\n--- Markdown Report ---")
    print(final_state.get("summary_md", ""))
    if final_state.get("justification_md"):
        print("\n--- Justification ---")
        print(final_state["justification_md"])
    print("\n--- Final Result ---")
    print(final_state.get("result", "No result"))
    import pprint
    pprint.pprint(final_state, width=120)
