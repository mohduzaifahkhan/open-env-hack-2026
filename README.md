---
title: Smart Factory Environment
emoji: 🏭
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# 🏭 TRIBUNAL — Smart Factory Assembly

> **AI-Powered Manufacturing RL Environment** • Meta × PyTorch × Scaler OpenEnv Hackathon 2026

[![OpenEnv](https://img.shields.io/badge/OpenEnv-v0.2.2-blue)](https://github.com/meta-pytorch/OpenEnv)
[![Python](https://img.shields.io/badge/Python-3.10%2B-green)](https://python.org)
[![HF Space](https://img.shields.io/badge/🤗-Live_Demo-yellow)](https://uzaif1-meta-hack-openenv-26.hf.space)
[![License](https://img.shields.io/badge/License-BSD--3-orange)](LICENSE)

**🔗 Live Demo: [uzaif1-meta-hack-openenv-26.hf.space](https://uzaif1-meta-hack-openenv-26.hf.space)**

Open the Space to see a **live visual demo** of the AI agent navigating the factory grid in real-time!

---

## 🌟 Key Features

| Feature | Description |
|---------|-------------|
| 🤖 **LLM-Powered Agent** | Every decision made by Llama-3.2-1B-Instruct via few-shot prompting |
| 🏗️ **Assembly Line Mechanic** | Parts must be delivered in correct sequence for bonus multipliers |
| 🔍 **Quality Inspection** | Stations produce defective parts (15%) — agent must manage risk |
| 📊 **Grading Rubric** | 5-metric weighted scoring system for agent evaluation |
| 🎲 **Stochastic Events** | Machine breakdowns and conveyor drift in Hard mode |
| 🖥️ **Visual Renderer** | Real-time animated web UI with live stats and AI thought display |
| 📈 **3 Difficulty Tiers** | Easy (5×5) → Medium (7×7) → Hard (10×10) with progressive complexity |

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                       Client / Agent Layer                       │
│  ┌──────────────────┐    ┌────────────────────────────────────┐  │
│  │ SmartFactoryEnv  │    │ inference.py (LLM Agent)           │  │
│  │ (WebSocket)      │    │ • Llama-3.2-1B via HF Router       │  │
│  │                  │    │ • Few-shot prompting                │  │
│  └────────┬─────────┘    │ • Conversation history              │  │
│           │              └────────────────┬───────────────────┘  │
└───────────┼───────────────────────────────┼──────────────────────┘
            │ WebSocket / HTTP              │
┌───────────▼───────────────────────────────▼──────────────────────┐
│              HuggingFace Spaces (Docker + FastAPI)                │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │  GET /           → Visual Renderer (animated grid)         │  │
│  │  GET /metrics    → Grading Rubric (JSON)                   │  │
│  │  WS  /ws/demo    → Live demo WebSocket                     │  │
│  │  WS  /ws         → Agent session (reset, step, state)      │  │
│  ├────────────────────────────────────────────────────────────┤  │
│  │          SmartFactoryEnvironment (OpenEnv)                 │  │
│  │  • Assembly line sequencing    • Quality inspection         │  │
│  │  • Grading rubric (5 metrics)  • Stochastic events         │  │
│  │  • Distance-based reward shaping • 3 difficulty tiers      │  │
│  └────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────┘
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
| `inventory` | `dict` | Parts still needed: `{"gear": 1}` |
| `time_remaining` | `int` | Steps before timeout |
| `deliveries_made` | `int` | Parts delivered so far |
| `deliveries_required` | `int` | Total to complete episode |
| `distance_to_target` | `float` | Manhattan distance to current target |
| `assembly_progress` | `str[]` | Parts assembled in order: `["gear", "chip"]` |
| `metadata.rubric` | `dict` | Live grading rubric scores |
| `metadata.inspection_results` | `dict` | Quality inspection results |

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
| 8  | INSPECT| Inspect station quality (reveals defective parts) |

### Difficulty Tiers

| Tier | Grid | Steps | Parts | Obstacles | Special |
|------|------|-------|-------|-----------|---------|
| 🟢 Easy   | 5×5  | 50  | 1 (gear) | None | — |
| 🟡 Medium | 7×7  | 75  | 2 (gear, chip) | 3 walls, 2 hazards | Ordered delivery |
| 🔴 Hard   | 10×10| 120 | 3 (gear, chip, frame) | 8 walls, 5 hazards, 3 conveyors | Stochastic events + Quality inspection |

### Reward Structure

| Event | Reward |
|-------|--------|
| Step cost | −0.05 |
| Distance progress | +0.1 × Δdistance |
| Grab correct part | +2.0 (×1.5 if correct assembly sequence) |
| Deliver part | +5.0 + speed bonus (up to +2.0) |
| Perfect assembly sequence | ×1.5 multiplier |
| Defective part delivery | +0.5 (reduced) |
| Wall collision | −1.0 |
| Hazard step | −0.5 |
| Idle action | −0.1 |
| Episode completion | +3.0 × efficiency × 10 |

### Grading Rubric

| Metric | Weight | Description |
|--------|--------|-------------|
| Completion Rate | 40% | deliveries_made / deliveries_required |
| Efficiency | 25% | Deliveries per steps taken |
| No Collisions | 15% | 1 − (collisions / steps) |
| Hazard Safety | 10% | 1 − (hazard_steps / steps) |
| Speed | 10% | Steps remaining when done / max_steps |

---

## 🚀 Quickstart

### 1. Live Demo (No Setup Required)

Visit **[uzaif1-meta-hack-openenv-26.hf.space](https://uzaif1-meta-hack-openenv-26.hf.space)** and click **"Run Demo"**.

### 2. Run Locally

```bash
# Clone
git clone https://github.com/mohduzaifahkhan/open-env-hack-2026.git
cd open-env-hack-2026

# Install
pip install -e .

# Start the environment server
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

### 3. Run the LLM Agent

```bash
# Set your HuggingFace API key
export API_KEY=your_hf_token_here    # Linux/Mac
set API_KEY=your_hf_token_here       # Windows

# Run across all 3 difficulty tiers
python inference.py
```

### 4. Use the Client Library

```python
from client import SmartFactoryEnv
from models import FactoryAction

async with SmartFactoryEnv(base_url="http://localhost:7860") as env:
    result = await env.reset(task="smart_factory_easy")
    print(result.observation.robot_pos)

    result = await env.step(FactoryAction(action=5))  # Move UP
    print(f"Reward: {result.reward}, Done: {result.done}")
    print(f"Rubric: {result.observation.metadata['rubric']}")
```

---

## 🧱 OpenEnv Compliance

This environment is fully compliant with `openenv-core>=0.2.2`:

- **Typed Models** → `models.py` with `FactoryAction(Action)`, `FactoryObservation(Observation)`, `FactoryState(State)`
- **Environment Subclass** → `server/environment.py` with `SmartFactoryEnvironment(Environment)`
- **Server App** → `server/app.py` using `create_app()` from `openenv.core.env_server.http_server`
- **Client** → `client.py` with `SmartFactoryEnv(EnvClient)` for WebSocket sessions
- **Manifest** → `openenv.yaml` with `spec_version: 1`

---

## 📁 Project Structure

```
open-env-hack-2026/
├── inference.py             # LLM agent (Llama-3.2-1B) with few-shot prompting
├── models.py                # FactoryAction, FactoryObservation, FactoryState
├── client.py                # SmartFactoryEnv(EnvClient) — WebSocket client
├── openenv.yaml             # OpenEnv manifest (spec_version: 1)
├── pyproject.toml           # Dependencies and packaging
├── Dockerfile               # HuggingFace Spaces deployment
├── requirements.txt         # Server dependencies
├── README.md                # This file
└── server/
    ├── environment.py        # SmartFactoryEnvironment with rubric + assembly
    ├── app.py                # FastAPI server with visual renderer + demo
    └── static/
        └── index.html        # Visual renderer (animated grid, live stats)
```

---

## 🧪 Example Output

```
============================================================
[START] task=smart_factory_easy
============================================================
🤖 AI: {"thought":"move UP toward pickup","action":5}
  [STEP] step=1 action=5 pos=[1, 2] carrying=0 deliveries=0/1
🤖 AI: {"thought":"move LEFT toward pickup","action":3}
  [STEP] step=2 action=3 pos=[1, 1] carrying=0 deliveries=0/1
...
🤖 AI: {"thought":"at pickup, grab","action":1}
  [STEP] step=5 action=1 reward=1.150 carrying=1 deliveries=0/1
...
🤖 AI: {"thought":"place at dropoff","action":2}
  [STEP] step=14 action=2 reward=7.733 carrying=0 deliveries=1/1

📊 FINAL SUMMARY — TRIBUNAL Smart Factory Agent
============================================================
Task                         Deliveries   Steps     Reward    Score
------------------------- ------------ ------- ---------- --------
smart_factory_easy                 1/1      14      9.483   0.9569
smart_factory_medium               1/2      45      4.521   0.3261
smart_factory_hard                 1/3      89      3.102   0.1458
------------------------- ------------ ------- ---------- --------
                        AVERAGE                              0.4763
Model: meta-llama/Llama-3.2-1B-Instruct
============================================================
```

---

## 🏆 What Makes This Special

1. **Genuine LLM Decision-Making** — Every action is decided by Llama-3.2-1B via few-shot prompting. No hardcoded navigation heuristics.

2. **Assembly Line Mechanic** — Not just pickup-deliver; parts must follow a manufacturing sequence (gear → chip → frame) for bonus rewards.

3. **Quality Inspection System** — 15% chance of defective parts on hard mode adds strategic depth to the INSPECT action.

4. **Built-in Grading Rubric** — 5-metric weighted scoring exposed via `/metrics` and in every observation, enabling automated evaluation.

5. **Live Visual Renderer** — Beautiful animated web UI showing the AI agent in real-time with thought bubbles and rubric scores.

---

## 📜 License

BSD-3-Clause — see [LICENSE](LICENSE) for details.

Built with ❤️ by **TRIBUNAL Team** for the Meta × PyTorch × Scaler OpenEnv Hackathon 2026.
