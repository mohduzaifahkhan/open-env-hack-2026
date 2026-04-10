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
    SYSTEM_PROMPT = f"""You are a simple robot on a grid. Follow these strict mathematical rules:

IF carrying=0 (Goal is y=0, x=0):
- Rule A: If y > 0 -> output UP [5]
- Rule B: If y == 0 and x > 0 -> output LEFT [3]
- Rule C: If y == 0 and x == 0 -> output GRAB [1]

IF carrying=1 (Goal is y={grid_max}):
- Rule D: If y < {grid_max} -> output DOWN [6]
- Rule E: If y == {grid_max} -> output PLACE [2]

OUTPUT FORMAT:
You must output in this exact sequence:
1. Explicitly state the current value of carrying.
2. Explicitly state the current value of y.
3. Explicitly state the current value of x.
4. Select the matching rule based strictly on those values.
5. Output the action in brackets.

EXAMPLES:
State: y=2, x=2, carrying=0, grid_max={grid_max}
AI: The current value of carrying is 0. The current value of y is 2. The current value of x is 2. carrying is 0 and y is > 0, so I follow Rule A. UP [5]

State: y=0, x=3, carrying=0, grid_max={grid_max}
AI: The current value of carrying is 0. The current value of y is 0. The current value of x is 3. carrying is 0, y is 0, and x is > 0, so I follow Rule B. LEFT [3]

State: y=0, x=0, carrying=0, grid_max={grid_max}
AI: The current value of carrying is 0. The current value of y is 0. The current value of x is 0. carrying is 0, y is 0, and x is 0, so I follow Rule C. GRAB [1]

State: y=0, x=0, carrying=1, grid_max={grid_max}
AI: The current value of carrying is 1. The current value of y is 0. The current value of x is 0. carrying is 1 and y is < {grid_max}, so I follow Rule D. DOWN [6]

State: y={grid_max}, x=0, carrying=1, grid_max={grid_max}
AI: The current value of carrying is 1. The current value of y is {grid_max}. The current value of x is 0. carrying is 1 and y is == {grid_max}, so I follow Rule E. PLACE [2]
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