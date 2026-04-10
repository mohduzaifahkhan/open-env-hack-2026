from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import uvicorn

app = FastAPI()

# Global state
state = {"robot_pos": [2, 2], "carrying": 0, "grid_max": 4, "steps": 0, "done": False}

def get_obs():
    return {
        "robot_pos": state.get("robot_pos", [0, 0]),
        "carrying": state.get("carrying", 0),
        "grid_max": state.get("grid_max", 4)
    }

@app.get("/")
async def root():
    return {"status": "online"}

@app.post("/reset")
async def reset_environment(request: Request):
    try:
        # Extra safety: check if body exists before parsing
        body = await request.json() if await request.body() else {}
    except Exception:
        body = {}

    ctx = body.get("ctx", {})
    # Use .get() everywhere to prevent KeyErrors
    task = str(ctx.get("task", "smart_factory_easy")).lower()

    if "hard" in task:
        grid_max = 9
    elif "medium" in task:
        grid_max = 6
    else:
        grid_max = 4

    state["robot_pos"] = [grid_max // 2, grid_max // 2]
    state["carrying"] = 0
    state["grid_max"] = grid_max
    state["steps"] = 0
    state["done"] = False

    return JSONResponse(content={
        "observation": get_obs(),
        "reward": 0.0,
        "done": False
    })

@app.post("/step")
async def step_environment(request: Request):
    try:
        body = await request.json()
    except Exception:
        body = {}
        
    # Standardize action extraction
    action_data = body.get("action", {})
    if isinstance(action_data, dict):
        action = action_data.get("action", 0)
    else:
        action = action_data

    state["steps"] += 1
    reward = -0.05 
    
    y, x = state["robot_pos"]
    carrying = state["carrying"]
    grid_max = state["grid_max"]

    if action == 3 and x > 0: x -= 1
    elif action == 4 and x < grid_max: x += 1
    elif action == 5 and y > 0: y -= 1
    elif action == 6 and y < grid_max: y += 1
    elif action == 1 and y == 0 and x == 0 and carrying == 0:
        carrying = 1
        reward += 2.0
    elif action == 2 and y == grid_max and carrying == 1:
        carrying = 0
        reward += 5.0
        state["done"] = True

    state["robot_pos"] = [y, x]
    state["carrying"] = carrying
    if state["steps"] >= 50: state["done"] = True

    return JSONResponse(content={
        "observation": get_obs(),
        "reward": reward,
        "done": state["done"]
    })

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
    # ... (all your existing FastAPI routes go above this) ...

def main():
    """Entry point for the OpenEnv multi-mode deployment validator."""
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)

if __name__ == '__main__':
    main()