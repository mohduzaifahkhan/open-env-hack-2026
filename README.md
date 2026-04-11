# 🏭 TRIBUNAL: Smart Factory Assembly Environment

**OpenEnv-compliant multi-difficulty manufacturing RL environment**
for the [Meta PyTorch OpenEnv Hackathon 2026](https://github.com/meta-pytorch/OpenEnv).

[![OpenEnv](https://img.shields.io/badge/OpenEnv-v0.2.2-blue)](https://github.com/meta-pytorch/OpenEnv)
[![Python](https://img.shields.io/badge/Python-3.10%2B-green)](https://python.org)
[![License](https://img.shields.io/badge/License-BSD--3-orange)](LICENSE)

---

## 🌟 Architecture Overview

```
┌──────────────────────────────────────────────────────────────┐
│                    Client Application                        │
│  ┌──────────────────┐    ┌──────────────────────────────┐   │
│  │ SmartFactoryEnv  │    │ inference.py (LLM Agent)     │   │
│  │ (EnvClient)      │    │ Llama-3.2-1B via HF Router   │   │
│  └────────┬─────────┘    └──────────────┬───────────────┘   │
└───────────┼─────────────────────────────┼───────────────────┘
            │ WebSocket / HTTP            │ HTTP POST
            │ (reset, step, state)        │ (/reset, /step)
┌───────────▼─────────────────────────────▼───────────────────┐
│              HuggingFace Spaces (Docker)                     │
│  ┌──────────────────────────────────────────────────────┐   │
│  │              FastAPI Server (server/app.py)           │   │
│  │  ┌────────────────────────────────────────────────┐  │   │
│  │  │     SmartFactoryEnvironment (Environment)      │  │   │
│  │  │  • Grid generation   • Reward shaping          │  │   │
│  │  │  • Obstacle logic    • Stochastic events       │  │   │
│  │  │  • Part management   • Quality constraints     │  │   │
│  │  └────────────────────────────────────────────────┘  │   │
│  └──────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────┘
```

---

## 📊 Environment Specification

### Observation Space

| Field | Type | Description |
|-------|------|-------------|
| `robot_pos` | `[y, x]` | Current robot coordinates |
| `carrying` | `0 \| 1` | Whether robot holds a part |
| `carrying_type` | `str?` | Part type: `"gear"`, `"chip"`, `"frame"` |
| `grid_layout` | `int[][]` | 2D grid (0=empty, 1=wall, 2=hazard, 3=pickup, 4=dropoff, 5=conveyor, 6=robot) |
| `inventory` | `dict` | Parts still needed: `{"gear": 1, "chip": 2}` |
| `time_remaining` | `int` | Steps before timeout |
| `deliveries_made` | `int` | Parts delivered so far |
| `deliveries_required` | `int` | Total to complete episode |
| `distance_to_target` | `float` | Manhattan distance to current target |

### Action Space (Discrete 0–8)

| ID | Action | Description |
|----|--------|-------------|
| 0  | NOOP   | Do nothing |
| 1  | GRAB   | Pick up part at pickup station |
| 2  | PLACE  | Deliver part at dropoff station |
| 3  | LEFT   | Move x−1 |
| 4  | RIGHT  | Move x+1 |
| 5  | UP     | Move y−1 |
| 6  | DOWN   | Move y+1 |
| 7  | WAIT   | Wait (conveyor tick) |
| 8  | INSPECT| Inspect cell quality |

### Difficulty Tiers

| Tier | Grid | Steps | Parts | Obstacles | Conveyors | Stochastic |
|------|------|-------|-------|-----------|-----------|------------|
| Easy   | 5×5  | 50  | 1 (gear) | None | No  | No |
| Medium | 7×7  | 75  | 2 (gear, chip) | 3 walls, 2 hazards | No  | No |
| Hard   | 10×10| 120 | 3 (gear, chip, frame) | 8 walls, 5 hazards | 3 conveyors | Yes (breakdowns) |

### Reward Structure

| Event | Reward |
|-------|--------|
| Step cost | −0.05 |
| Distance progress | +0.1 × Δdistance |
| Grab correct part | +2.0 (×0.5 if wrong order) |
| Deliver part | +5.0 + speed bonus (up to +2.0) |
| Wall collision | −1.0 |
| Hazard step | −0.5 |
| Idle action | −0.1 |
| Episode completion | +3.0 × efficiency |

---

## 🧱 OpenEnv Compliance

This environment is built on the `openenv-core>=0.2.2` framework:

- **Typed Models** → `models.py` with `FactoryAction(Action)`, `FactoryObservation(Observation)`, `FactoryState(State)`
- **Environment Subclass** → `server/environment.py` with `SmartFactoryEnvironment(Environment)`
- **Server App** → `server/app.py` using `create_app()` from `openenv.core.env_server.http_server`
- **Client** → `client.py` with `SmartFactoryEnv(EnvClient)` for WebSocket sessions
- **Manifest** → `openenv.yaml` with `spec_version: 1`

---

## 🚀 How to Run

### Prerequisites

```bash
pip install openenv-core[core]>=0.2.2 openai requests
```

### 1. The Environment Server (Already Live)

The environment is deployed at: `https://uzaif1-meta-hack-openenv-26.hf.space`

To run locally:

```bash
git clone https://github.com/mohduzaifahkhan/open-env-hack-2026.git
cd open-env-hack-2026
pip install -e .
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

### 2. Run the Inference Agent

```bash
# Set your HuggingFace API key
# Linux/Mac:
export API_KEY=your_hf_token_here

# Windows:
set API_KEY=your_hf_token_here

# Run the agent
python inference.py
```

The agent will automatically execute all 3 difficulty tiers and output normalized scores.

### 3. Using the Client Library

```python
from client import SmartFactoryEnv
from models import FactoryAction

# Sync usage
with SmartFactoryEnv(base_url="http://localhost:7860").sync() as env:
    result = env.reset(task="smart_factory_easy")
    print(result.observation.robot_pos)

    result = env.step(FactoryAction(action=5))  # Move UP
    print(f"Reward: {result.reward}, Done: {result.done}")
```

---

## 📁 Project Structure

```
open-env-hack-2026/
├── __init__.py              # Package exports
├── models.py                # FactoryAction, FactoryObservation, FactoryState
├── client.py                # SmartFactoryEnv(EnvClient) — WebSocket client
├── inference.py             # LLM inference agent (Llama-3.2-1B)
├── openenv.yaml             # OpenEnv manifest (spec_version: 1)
├── pyproject.toml           # Dependencies and packaging
├── Dockerfile               # HuggingFace Spaces deployment
├── requirements.txt         # Server dependencies
├── README.md                # This file
└── server/
    ├── __init__.py           # Server package
    ├── environment.py        # SmartFactoryEnvironment(Environment)
    └── app.py                # FastAPI app using create_app()
```

---

## 🧪 Example Output

```
============================================================
[START] task=smart_factory_easy
============================================================
🤖 AI: {"thought": "Not carrying, at (2,2), pickup at (0,0). Move UP.", "action": 5}
  [STEP] step=1 action=5 reward=0.145 total=0.145 pos=[1, 2] carrying=0
🤖 AI: {"thought": "Continue UP toward pickup at (0,0).", "action": 5}
  [STEP] step=2 action=5 reward=0.145 total=0.290 pos=[0, 2] carrying=0
...
📊 Results: total_reward=8.742 steps=12
[END] task=smart_factory_easy score=0.916 steps=12
```

---

## 📜 License

BSD-3-Clause — see [LICENSE](LICENSE) for details.

Built with ❤️ for the Meta PyTorch OpenEnv Hackathon 2026.