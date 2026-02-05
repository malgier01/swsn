import json
import time
import concurrent.futures
import pandas as pd
import google.generativeai as genai
from google.generativeai.types import content_types
import os
import sys

# Konfiguracja API
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
if not GOOGLE_API_KEY:
    print("Error: GOOGLE_API_KEY environment variable not set.")
    sys.exit(1)

genai.configure(api_key=GOOGLE_API_KEY)
print("Gemini API configured globally.")

MAX_WORKERS = 3

# LLM JUDGE
def run_judgment_llm(criteria_name, criteria_desc, content):
    """
    Evaluates text and extracts a knowledge graph of the reasoning.
    """
    judge_model = genai.GenerativeModel('gemini-2.5-flash')

    prompt = f"""
    ROLE: Specialized Evaluator for {criteria_name}.
    TASK: Analyze the "Content" against the "Criteria".

    CRITERIA DESCRIPTION: {criteria_desc}
    CONTENT: {content}

    OUTPUT FORMAT: Return a valid JSON object with:
    1. "score": 1 (Pass) or 0 (Fail).
    2. "explanation": A brief explanation.
    3. "graph_nodes": A list of objects {{"id": "concept_or_claim", "type": "tag"}}.
       - Extract key concepts from the text relevant to the critique.
    4. "graph_edges": A list of objects {{"source": "id", "target": "id", "relationship": "verb"}}.
       - Map the logic. E.g., "Input Text" -> "LACKS" -> "Empathy".
    """

    try:
        response = judge_model.generate_content(
            prompt,
            generation_config={"response_mime_type": "application/json"}
        )
        return json.loads(response.text)
    except Exception as e:
        return {
            "score": 0,
            "explanation": f"Judge crashed: {e}",
            "graph_nodes": [],
            "graph_edges": []
        }

# TOOL CREATION (Placeholders for tool calls)
def check_accuracy(text_content: str):
    """Validates if the text is factually correct."""
    return "This is a placeholder. The logic happens in handle_tool_call."

def check_completeness(text_content: str):
    """Checks if the text answers all parts comprehensively."""
    return "This is a placeholder. The logic happens in handle_tool_call."

def check_empathy(text_content: str):
    """Checks if the tone is appropriate and supportive."""
    return "This is a placeholder. The logic happens in handle_tool_call."

# AGENT FACTORY
def create_agent(agent_name, tool_function, criteria_desc):

    tools_for_model = [tool_function]

    system_instruction = f"""
    ROLE: {agent_name} Validator.
    TASK: You are a classifier.
    1. Receive user text.
    2. IMMEDIATELY call the function '{tool_function.__name__}'.
    3. Pass the user text into the tool's 'text_content' argument.
    """

    chat_model = genai.GenerativeModel(
        'gemini-2.5-flash',
        tools=tools_for_model,
        system_instruction=system_instruction
    )

    def handle_tool_call(tool_call, context_text):
        print(f"   >>> {agent_name} is judging...")

        input_text = context_text
        if tool_call.args and 'text_content' in tool_call.args:
            input_text = tool_call.args['text_content']

        result = run_judgment_llm(
            criteria_name=agent_name,
            criteria_desc=criteria_desc,
            content=input_text
        )

        print(f"   >>> {agent_name} Score: {result.get('score')}")

        return {
            "status": "success",
            "judgment_score": result.get('score'),
            "judgment_reason": result.get('explanation'),
            "graph_nodes": result.get('graph_nodes', []),
            "graph_edges": result.get('graph_edges', [])
        }

    return chat_model, handle_tool_call

# THREADED WORKER
def run_agent_threaded(agent_name, agent_model, handle_tool_call, user_query):

    chat = agent_model.start_chat(history=[])

    forced_mode = content_types.to_tool_config({"function_calling_config": {"mode": "ANY"}})
    auto_mode = content_types.to_tool_config({"function_calling_config": {"mode": "AUTO"}})

    final_data = {
        f"{agent_name}_grade": 0,
        f"{agent_name}_reason": "Failed to run",
        f"{agent_name}_graph": "[]"
    }

    try:
        response = chat.send_message(user_query, tool_config=forced_mode)

        if response.candidates and response.candidates[0].content.parts:
            part = response.candidates[0].content.parts[0]

            if part.function_call:

                result = handle_tool_call(part.function_call, context_text=user_query)

                if isinstance(result, dict):
                    final_data[f"{agent_name}_grade"] = result.get('judgment_score', 0)
                    final_data[f"{agent_name}_reason"] = result.get('judgment_reason', '')

                    graph_data = {
                        "nodes": result.get('graph_nodes'),
                        "edges": result.get('graph_edges')
                    }
                    final_data[f"{agent_name}_graph"] = json.dumps(graph_data)

                tool_resp = genai.protos.Part(
                    function_response=genai.protos.FunctionResponse(
                        name=part.function_call.name,
                        response={'result': result}
                    )
                )
                chat.send_message([tool_resp], tool_config=auto_mode)

    except Exception as e:
        final_data[f"{agent_name}_reason"] = f"Error: {str(e)}"
        print(f" {agent_name} crashed: {e}")

    return final_data

# ORCHESTRATOR
def process_single_row_threaded(row, active_agents_dict):
    answer_text = row.get('Answer')
    answer_id = row.get('id', 'Unknown')

    print(f" [ID: {answer_id}] Starting parallel evaluation...")

    results_to_merge = {}

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_agent = {}

        for name, (model, handler) in active_agents_dict.items():
            future = executor.submit(
                run_agent_threaded,
                name,
                model,
                handler,
                answer_text
            )
            future_to_agent[future] = name

        for future in concurrent.futures.as_completed(future_to_agent):
            try:
                data = future.result()
                results_to_merge.update(data)
            except Exception as exc:
                print(f"Thread exc: {exc}")

    combined = row.copy()
    combined.update(results_to_merge)
    print(f" [ID: {answer_id}] Finished.")
    return combined

def run_parallel_system_threaded(csv_path):
    try:
        df = pd.read_csv(csv_path)
    except:
        print("CSV not found, using dummy data.")
        df = pd.DataFrame([
            {"id": 101, "Answer": "The sun is cold and blue."},
            {"id": 102, "Answer": "I understand this is difficult. The treatment is effective."}
        ])

    agent_configs = [
        ("Accuracy", check_accuracy, "Factually accurate, citing correct numbers."),
        ("Completeness", check_completeness, "Addresses every aspect of the question."),
        ("Empathy", check_empathy, "Tone is warm, understanding, and human-like.")
    ]

    active_agents = {}
    print(" Initializing Agents...")
    for name, func, desc in agent_configs:
        model, handler = create_agent(name, func, desc)
        active_agents[name] = (model, handler)

    final_rows = []
    for index, row in df.iterrows():
        row_dict = row.to_dict()
        if 'id' not in row_dict: row_dict['id'] = index

        enriched_row = process_single_row_threaded(row_dict, active_agents)
        final_rows.append(enriched_row)

        time.sleep(1)

    output_filename = "FINAL_threaded_report_with_graphs.csv"
    pd.DataFrame(final_rows).to_csv(output_filename, index=False)
    print(f"Done! Saved to {output_filename}")

# EXECUTE
if __name__ == "__main__":
    # Upewnij się, że plik wejściowy istnieje
    if not os.path.exists("sm_answers.csv"):
        print("Brak pliku sm_answers.csv. Uruchom najpierw skrypt generujący dane.")
    else:
        run_parallel_system_threaded("sm_answers.csv")