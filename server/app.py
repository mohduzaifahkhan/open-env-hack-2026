from fastapi import FastAPI
import numpy as np
from pydantic import BaseModel, Field
from typing import Dict, Any, Tuple
from openenv.core.env_server import Environment

class RobotAction(BaseModel):
    action: int = Field(
        ge=0, le=6, 
        description="0: NOOP, 1: GRAB, 2: PLACE, 3: LEFT, 4: RIGHT, 5: UP, 6: DOWN"
    )

class FactoryObservation(BaseModel):
    grid: list[list[int]] = Field(description="5x5 numpy int8 Grid representing the factory floor")
    robot_pos: list[int] = Field(description="Robot current coordinates [y, x]")
    carrying: int = Field(description="Part type currently held by the robot (0 if empty)")
    metrics: Dict[str, float] = Field(default_factory=dict, description="Internal Grader Metrics")

class SmartFactoryEnv(Environment):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.grid_size = 5
        self.max_steps = 100
        self.current_step = 0
        
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int8)
        self.grid[0, 0] = 1 # Bin A Origin
        self.grid[0, 4] = 2 # Bin B Origin
        self.grid[4, :] = 3 # Continuous conveyor row
        
        self.robot_pos = np.array([2, 2], dtype=np.int8) 
        self.carrying = np.int8(0)
        self.sequence_needed = np.array([1, 2, 1, 2], dtype=np.int8) 
        self.placed_parts = []
        
        self.parts_placed = 0
        self.sequence_breaches = 0

    def reset(self, ctx: dict) -> FactoryObservation:
        self.current_step = 0
        self.robot_pos = np.array([2, 2], dtype=np.int8)
        self.carrying = np.int8(0)
        self.placed_parts = []
        self.parts_placed = 0
        self.sequence_breaches = 0
        return self._get_obs()

    def _get_obs(self) -> FactoryObservation:
        metrics = {
            "parts_placed": float(self.parts_placed),
            "sequence_breaches": float(self.sequence_breaches)
        }
        return FactoryObservation(
            grid=self.grid.tolist(),
            robot_pos=self.robot_pos.tolist(),
            carrying=int(self.carrying),
            metrics=metrics
        )

    def step(self, action: RobotAction, ctx: dict) -> Tuple[FactoryObservation, float, bool, Dict[str, Any]]:
        self.current_step += 1
        reward = -0.01 
        done = False
        
        act = action.action
        move_vectors = {
            3: np.array([0, -1], dtype=np.int8), 
            4: np.array([0, 1], dtype=np.int8),  
            5: np.array([-1, 0], dtype=np.int8), 
            6: np.array([1, 0], dtype=np.int8)   
        }
        
        if 3 <= act <= 6:
            new_pos = self.robot_pos + move_vectors[act]
            new_pos = np.clip(new_pos, 0, self.grid_size - 1)
            
            if np.array_equal(self.robot_pos, new_pos):
                reward -= 0.5 
            else:
                self.robot_pos = new_pos
                
        elif act == 1: 
            cell_type = self.grid[self.robot_pos[0], self.robot_pos[1]]
            if self.carrying == 0 and (cell_type == 1 or cell_type == 2):
                self.carrying = cell_type 
            else:
                reward -= 0.5 
                
        elif act == 2: 
            cell_type = self.grid[self.robot_pos[0], self.robot_pos[1]]
            if self.carrying != 0 and cell_type == 3: 
                self.placed_parts.append(self.carrying)
                self.parts_placed += 1
                reward += 1.0 
                
                idx = len(self.placed_parts) - 1
                if idx < len(self.sequence_needed):
                    if self.placed_parts[idx] == self.sequence_needed[idx]:
                        reward += 10.0 
                    else:
                        self.sequence_breaches += 1
                else:
                    self.sequence_breaches += 1 

                self.carrying = np.int8(0) 
            else:
                reward -= 0.5 

        if self.current_step >= self.max_steps:
            done = True
        elif len(self.placed_parts) >= len(self.sequence_needed):
            done = True 

        return self._get_obs(), float(reward), done, {}

    def state(self, ctx: dict) -> FactoryObservation:
        return self._get_obs()

# 1. Create the Environment instance
env = SmartFactoryEnv()

# 2. Build the FastAPI app manually
app = FastAPI()

# 3. Create the endpoints the inference script is looking for
@app.post("/reset")
async def reset(ctx: dict = {}):
    return env.reset(ctx)

@app.post("/step")
async def step(data: dict):
    # This maps the incoming JSON to the RobotAction model
    action_obj = RobotAction(action=data["action"]["action"])
    obs, reward, done, info = env.step(action_obj, data.get("ctx", {}))
    return {"observation": obs, "reward": reward, "done": done, "info": info}

@app.get("/state")
async def state(ctx: dict = {}):
    return env.state(ctx)
def main():
    import uvicorn
    # This makes the main() function callable by the validator
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()