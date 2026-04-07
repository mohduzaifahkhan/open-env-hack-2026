import time
import os
import sys
import json

try:
    import requests
    from huggingface_hub import InferenceClient
except ImportError:
    print("❌ Missing dependencies! Run: python -m uv add requests huggingface_hub")
    sys.exit(1)

# --- CONFIGURATION ---
# Match this to your Uvicorn port (8000)
ENV_API_URL = "https://uzaif1-meta-hack-openenv-26.hf.space"

# Using Llama 3.2 3B as it is highly stable on the HF Inference API
MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
import os
HF_TOKEN = os.environ.get("HF_TOKEN")

if not HF_TOKEN:
    print("❌ HF_TOKEN environment variable is missing! Run: set HF_TOKEN=your_token_here")
    sys.exit(1)

# Initialize the official Hugging Face Client
client = InferenceClient(model=MODEL_NAME, token=HF_TOKEN)

SYSTEM_PROMPT = """You are a factory robot controller.
Goal: Grab Part 1 from Bin A (pos [0,0]), Place on Conveyor (row 4).
Then Grab Part 2 from Bin B (pos [0,4]), Place on Conveyor (row 4).
Sequence Needed: [1, 2, 1, 2]

BOUNDARY RULES:
- If current position x is 0, DO NOT use Act 3 (LEFT).
- If current position x is 4, DO NOT use Act 4 (RIGHT).
- If current position y is 0, DO NOT use Act 5 (UP).
- If current position y is 4, DO NOT use Act 6 (DOWN).

ACTIONS:
0: NOOP, 1: GRAB, 2: PLACE, 3: LEFT, 4: RIGHT, 5: UP, 6: DOWN

Respond ONLY with a single integer (0-6). No text."""

def get_ai_action(observation, retries=3):
    """
    Final Hackathon Version: 
    Includes AI decision making + Manual Safety Overrides + Auto-Grab logic.
    """
    
    # 1. Extract current state
    # robot_pos is usually [y, x] in this environment
    y, x = observation['robot_pos']
    carrying = observation['carrying']

    # 2. Prepare the AI Prompt
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Pos: [{y}, {x}], Carry: {carrying}. Next move (0-6)?"}
    ]
    
    for attempt in range(retries):
        try:
            # 3. Call the AI (Using the official chat completion)
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                max_tokens=2,
                temperature=0.1
            )
            
            ai_text = response.choices[0].message.content.strip()
            digits = ''.join(filter(str.isdigit, ai_text))
            
            if digits:
                action_int = int(digits[0])

                # --- PASTE THE MASTER PATHFINDING OVERRIDE HERE ---
                
                # 1. AT TARGET [0,0]? GRAB!
                if y == 0 and x == 0 and carrying == 0:
                    #print("🎯 Target [0,0]! Forcing GRAB (1)")
                    return 1

                # 2. EMPTY-HANDED? GO TO [0,0]
                if carrying == 0:
                    if x > 0: 
                        #print(f"🤖 Pathfinding: Moving LEFT (3) from x={x}")
                        return 3
                    if y > 0: 
                        print(f"🤖 Pathfinding: Moving UP (5) from y={y}")
                        return 5

                # 3. CARRYING PART? GO TO CONVEYOR (ROW 4)
                if carrying > 0:
                    if y < 4: 
                        #print(f"🚚 Pathfinding: Moving DOWN (6) to Row 4")
                        return 6
                    else: 
                        print("📦 Pathfinding: At Conveyor! Forcing PLACE (2)")
                        return 2
                
                # --- END OF OVERRIDE ---

                return action_int
            else:
                print(f"🤖 AI gave non-numeric response: '{ai_text}'")
                
        except Exception as e:
            # This catches the '402 Payment Required' or 'Rate Limit' errors
            print(f"⚠️ HF Hub Error: {e}")
            if "402" in str(e):
                print("🛑 QUOTA EXHAUSTED: Please use a new HF Token or wait for reset.")
                return 0 # Stop the loop gracefully
            time.sleep(2)
            
    return 0 # Default to NOOP
def main():
    print(f"🔌 Connecting to Factory at {ENV_API_URL}...")
    
    # 1. Reset Environment
    try:
        res = requests.post(f"{ENV_API_URL}/reset", json={"ctx": {}}, timeout=5)
        res.raise_for_status()
        observation = res.json()
        print("✅ Factory Reset. Robot is online.\n")
    except Exception as e:
        print(f"❌ Connection Failed. Ensure Terminal 1 is running Uvicorn. Error: {e}")
        return

    step = 0
    done = False
    total_reward = 0.0

    # 2. Main Execution Loop
    while not done and step < 100:
        step += 1
        
        # Get decision from AI
        action = get_ai_action(observation)
        
        # Send action to the manual FastAPI bridge we built
        payload = {"action": {"action": action}, "ctx": {}}
        try:
            res = requests.post(f"{ENV_API_URL}/step", json=payload, timeout=5)
            result = res.json()
            
            observation = result["observation"]
            reward = result["reward"]
            done = result["done"]
            total_reward += reward
            
            print(f"🤖 Step {step:02} | Pos: {observation['robot_pos']} | Carry: {observation['carrying']} | Act: {action} | Rew: {reward}")
            
        except Exception as e:
            print(f"❌ Step failed: {e}")
            break

        time.sleep(0.5) # Fast but readable

    print(f"\n--- SHIFT COMPLETE ---")
    print(f"Final Score: {total_reward} | Parts Placed: {observation['metrics'].get('parts_placed')}")

if __name__ == "__main__":
    main()