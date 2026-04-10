import os
import sys
import time
import requests
import re
from openai import OpenAI

# --- CONFIGURATION ---
API_BASE_URL = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1/")
API_KEY = os.environ.get("API_KEY")
MODEL_NAME = os.environ.get("MODEL_NAME", "meta-llama/Llama-3.2-1B-Instruct")
ENV_API_URL = os.environ.get("ENV_API_URL", "https://uzaif1-meta-hack-openenv-26.hf.space")

if not API_KEY:
    print("❌ ERROR: API_KEY is missing! Run: set API_KEY=your_key")
    sys.exit(1)

client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

def get_ai_action(observation):
    y, x = observation['robot_pos']
    carrying = observation['carrying']
    grid_max = observation['grid_max'] # Dynamically injected based on difficulty
    
    # --- DYNAMIC FEW-SHOT PROMPT ---
    # --- DYNAMIC DECISION TREE PROMPT ---
    SYSTEM_PROMPT = """You are a precise robotic controller. You must strictly evaluate your current state (y, x, carrying) against the following Decision Tree.

DECISION TREE:
Rule 1: If carrying == 0 AND y > 0 -> Output [5]
Rule 2: If carrying == 0 AND y == 0 AND x > 0 -> Output [3]
Rule 3: If carrying == 0 AND y == 0 AND x == 0 -> Output [1]
Rule 4: If carrying == 1 AND y < grid_max -> Output [6]
Rule 5: If carrying == 1 AND y == grid_max -> Output [2]

You MUST format your response exactly like the example below. Do not write sentences.

EXAMPLE OUTPUT:
y=0
x=2
carrying=0
Rule 1: False
Rule 2: True
Action: [3]
"""

    user_state = f"State: y={y}, x={x}, carrying={carrying}, grid_max={grid_max}\nAI:"

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_state}
            ],
            max_tokens=200,
            temperature=0.01
        )
        ai_response = response.choices[0].message.content.strip()
        print(f"🤖 AI: {ai_response}")
        
        # Extract the number in brackets
        match = re.search(r'\[([0-6])\]', ai_response)
        if match: return int(match.group(1))
            
        digits = ''.join(filter(str.isdigit, ai_response))
        return int(digits[-1]) if digits else 0
        
    except Exception as e:
        print(f"⚠️ LLM Error: {e}")
        return 0 

def main():
    # --- Handshake ---
    try: client.chat.completions.create(model=MODEL_NAME, messages=[{"role": "user", "content": "handshake"}], max_tokens=1)
    except: pass 
    
    # --- The 3 Difficulty Levels ---
    tasks = ["smart_factory_easy", "smart_factory_medium", "smart_factory_hard"]
    
    for task_name in tasks:
        print(f"\n[START] task={task_name}", flush=True)

        try:
            # Tell the HF space which difficulty to build
            payload = {"ctx": {"task": task_name}}
            res = requests.post(f"{ENV_API_URL}/reset", json=payload, timeout=10)
            res.raise_for_status()
            observation = res.json()["observation"]
        except Exception as e:
            print(f"Failed to reset: {e}")
            print(f"[END] task={task_name} score=0.01 steps=0", flush=True)
            continue

        step_count = 0
        done = False
        total_reward = 0.0

        # Max 50 steps to prevent infinite loops
        while not done and step_count < 50: 
            step_count += 1
            action = get_ai_action(observation)
            
            try:
                res = requests.post(f"{ENV_API_URL}/step", json={"action": {"action": action}, "ctx": {}}, timeout=10)
                result = res.json()
                
                observation = result["observation"]
                reward = result["reward"]
                done = result.get("done", False)
                total_reward += reward
                
                print(f"[STEP] step={step_count} reward={reward}", flush=True)
            except Exception as e:
                print(f"API Error: {e}")
                break

            time.sleep(0.05) 

        # Score Normalization (assuming max possible reward is roughly ~7.0)
        raw_scaled_score = (total_reward + 5.0) / 12.0 
        normalized_score = max(0.01, min(0.99, raw_scaled_score))
        print(f"[END] task={task_name} score={normalized_score} steps={step_count}", flush=True)

if __name__ == "__main__":
    main()