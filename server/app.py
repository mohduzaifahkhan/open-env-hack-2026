from fastapi import FastAPI, Request
import uvicorn

app = FastAPI()

# Global state to keep track of the robot during the episode
state = {}

def get_obs():
    return {
        "robot_pos": state["robot_pos"],
        "carrying": state["carrying"],
        "grid_max": state["grid_max"]
    }
@app.get("/")
async def root():
    return {
        "status": "online", 
        "message": "Smart Factory Environment API is running. Ready for /reset and /step commands."
    }
@app.post("/reset")
async def reset_environment(request: Request):
    data = await request.json()
    ctx = data.get("ctx", {})
    task = ctx.get("task", "smart_factory_easy")

    # Set difficulty based on the task requested by the client
    if task == "smart_factory_hard":
        grid_max = 9  # 10x10 grid (0 to 9)
    elif task == "smart_factory_medium":
        grid_max = 6  # 7x7 grid (0 to 6)
    else:
        grid_max = 4  # 5x5 grid (0 to 4)

    # Reset the robot to the middle of the grid
    state["robot_pos"] = [grid_max // 2, grid_max // 2]
    state["carrying"] = 0
    state["grid_max"] = grid_max
    state["steps"] = 0
    state["done"] = False

    return {"observation": get_obs(), "reward": 0.0, "done": False}

@app.post("/step")
async def step_environment(request: Request):
    data = await request.json()
    action = data.get("action", {}).get("action", 0)

    state["steps"] += 1
    reward = -0.05  # Penalty for every step taken to encourage speed
    
    y, x = state["robot_pos"]
    carrying = state["carrying"]
    grid_max = state["grid_max"]

    # Process Movement Actions
    if action == 3 and x > 0: 
        x -= 1 # LEFT
    elif action == 4 and x < grid_max: 
        x += 1 # RIGHT
    elif action == 5 and y > 0: 
        y -= 1 # UP
    elif action == 6 and y < grid_max: 
        y += 1 # DOWN
        
    # Process GRAB Action (Must be at [0,0] and empty-handed)
    elif action == 1 and y == 0 and x == 0 and carrying == 0:
        carrying = 1
        reward += 2.0  # Big reward for grabbing
        
    # Process PLACE Action (Must be at bottom row and carrying part)
    elif action == 2 and y == grid_max and carrying == 1:
        carrying = 0
        reward += 5.0  # Massive reward for finishing
        state["done"] = True

    # Save new state
    state["robot_pos"] = [y, x]
    state["carrying"] = carrying

    # Force end if taking too long
    if state["steps"] >= 50:
        state["done"] = True

    return {"observation": get_obs(), "reward": reward, "done": state["done"]}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)