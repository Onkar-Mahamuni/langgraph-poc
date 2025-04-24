from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
import pandas as pd
from prophet import Prophet
import random
import time
from typing import TypedDict, Literal, Optional
from langgraph.graph import StateGraph, END
import os

# ---- Define the state schema ----
class ScaleMancerState(TypedDict, total=False):
    query: str
    cpu_df: Optional[object]
    user_df: Optional[object]
    cpu_forecast: Optional[object]
    user_forecast: Optional[object]
    avg_cpu: Optional[float]
    avg_users: Optional[float]
    cost: Optional[float]
    analysis_text: Optional[str]
    suggestion: Optional[Literal["up", "down", "same"]]
    confirmed: Optional[bool]
    result: Optional[str]


# ---------- LLM Setup (LangChain v0.3 / langchain_ollama) ----------
# llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=os.getenv("OPENAI_API_KEY"))
llm = ChatOpenAI(model="llama3.1:8b", temperature=0.1)

# ---------- Dummy External Interaction Functions ----------

def fetch_cpu_usage_data(start_time, end_time):
    print("Fetching CPU usage data...")
    dates = pd.date_range(start=start_time, end=end_time, freq="D")
    values = [random.uniform(30, 90) for _ in range(len(dates))]
    return pd.DataFrame({"timestamp": dates, "cpu_usage": values})

def fetch_active_customers_data(start_time, end_time):
    print("Fetching active customer data...")
    dates = pd.date_range(start=start_time, end=end_time, freq="D")
    values = [random.randint(500, 2000) for _ in range(len(dates))]
    return pd.DataFrame({"timestamp": dates, "active_customers": values})

def forecast_with_prophet(df, column_name):
    print(f"Forecasting {column_name} using Prophet...")
    df = df.rename(columns={"timestamp": "ds", column_name: "y"})
    model = Prophet()
    model.fit(df)
    future = model.make_future_dataframe(periods=7)
    forecast = model.predict(future)
    return forecast[["ds", "yhat"]]

def trigger_scaling_action(suggestion):
    print(f"Triggering scaling action: {suggestion.upper()} (simulated API call)")
    time.sleep(2)
    return f"Scaling action '{suggestion}' successfully triggered."

# ---------- LangGraph Nodes ----------

def intent_detection_node(state):
    print("Detecting intent using LLM...")
    query = state['query']
    prompt = f"""
    You are a system analyst. Classify the user query below into one of the following categories:
    - simple: If it's asking for past data (e.g., outages, performance)
    - normal: If it's asking for analysis or scale suggestion based on current trends
    - complex: If it's asking for optimization, future planning, or includes hypothetical inputs like user growth

    Query: {query}
    Category:
    """
    category = llm.invoke(prompt).content.strip().lower()
    if "simple" in category:
      return {"intent": "simple"}
    elif "complex" in category:
        return {"intent": "complex"}
    else:
        return {"intent": "normal"}

def fetch_data_node(state):
    print("Fetching data for forecasting...")
    cpu_df = fetch_cpu_usage_data("2024-01-01", "2024-01-31")
    user_df = fetch_active_customers_data("2024-01-01", "2024-01-31")
    return {**state, "cpu_df": cpu_df, "user_df": user_df}

def forecast_node(state):
    print("Running forecasting models...")
    cpu_forecast = forecast_with_prophet(state["cpu_df"], "cpu_usage")
    user_forecast = forecast_with_prophet(state["user_df"], "active_customers")
    return {**state, "cpu_forecast": cpu_forecast, "user_forecast": user_forecast}

def analysis_node(state):
    print("Performing analysis using LLM...")
    avg_cpu = state["cpu_forecast"]["yhat"].mean()
    avg_users = state["user_forecast"]["yhat"].mean()
    cost_est = avg_cpu * 0.05 + avg_users * 0.01

    prompt = f"""
    Analyze the following:
    - Average CPU usage forecast: {avg_cpu:.2f}%
    - Average active users forecast: {avg_users:.2f}
    - Estimated Azure VM cost: ${cost_est:.2f}

    Give an explanation of what this means and whether system is under pressure.
    """
    analysis = llm.invoke(prompt).content.strip()  # Access the 'content' attribute
    return {**state, "analysis_text": analysis, "avg_cpu": avg_cpu, "avg_users": avg_users, "cost": cost_est}

def decision_node(state):
    print("Deciding action using LLM...")
    prompt = f"""
    Based on the following inputs:
    - Avg CPU usage: {state['avg_cpu']:.2f}
    - Avg active users: {state['avg_users']:.2f}
    - Cost: ${state['cost']:.2f}

    Decide whether to scale up, scale down, or keep the system as-is.
    Only return one word: up, down, same.
    """
    suggestion = llm.invoke(prompt).content.strip().lower()
    return {**state, "suggestion": suggestion}

def human_in_loop_node(state):
    print("Waiting for human confirmation...")
    confirmation = input(f"Suggested action is '{state['suggestion']}'. Confirm? (yes/no): ")
    return {**state, "confirmed": confirmation.lower() == "yes"}

def action_node(state):
    if state["confirmed"]:
        result = trigger_scaling_action(state["suggestion"])
        return {**state, "result": result}
    else:
        print("Action not confirmed. Exiting without triggering scaling.")
        return {**state, "result": "No action taken."}

# ---------- Build the LangGraph ----------
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
    "simple": END,
    "normal": "FetchData",
    "complex": "FetchData"
})
graph.add_edge("FetchData", "Forecast")
graph.add_edge("Forecast", "Analyze")
graph.add_edge("Analyze", "Decide")
graph.add_edge("Decide", "Confirm")
graph.add_edge("Confirm", "Act")
graph.set_finish_point("Act")

app = graph.compile()

# ---------- Run the System ----------
if __name__ == '__main__':
    user_input = input("Enter your query: ")
    final_state = app.invoke({"query": user_input})
    print("\nFinal Output:", final_state.get("result", "No output"))
