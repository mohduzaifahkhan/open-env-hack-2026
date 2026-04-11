"""Quick smoke test for the Smart Factory environment and server."""
import threading
import time
import requests
import uvicorn

# -----------------------------------------------------------------------
# Test 1: Direct environment test (no server)
# -----------------------------------------------------------------------
print("=" * 60)
print("TEST 1: Direct Environment Test")
print("=" * 60)

from server.environment import SmartFactoryEnvironment
from models import FactoryAction, FactoryObservation, FactoryState

for task_name in ["smart_factory_easy", "smart_factory_medium", "smart_factory_hard"]:
    env = SmartFactoryEnvironment()
    obs = env.reset(task=task_name)
    print(f"\n  {task_name}:")
    print(f"    Grid: {len(obs.grid_layout)}x{len(obs.grid_layout[0])}")
    print(f"    Start: {obs.robot_pos}, Time: {obs.time_remaining}")
    print(f"    Deliveries needed: {obs.deliveries_required}")
    print(f"    Inventory: {obs.inventory}")

    # Run a few steps with heuristic
    for _ in range(5):
        obs = env.step(FactoryAction(action=5))  # UP

    st = env.state
    print(f"    After 5 steps: reward={st.total_reward:.3f}, state_steps={st.step_count}")

    # Check types
    assert isinstance(obs, FactoryObservation), "Observation type mismatch"
    assert isinstance(st, FactoryState), "State type mismatch"
    assert isinstance(env.get_metadata().name, str), "Metadata error"
    print(f"    ✅ Types correct")

print("\n✅ TEST 1 PASSED: All environments work directly\n")

# -----------------------------------------------------------------------
# Test 2: FastAPI server test
# -----------------------------------------------------------------------
print("=" * 60)
print("TEST 2: HTTP Server Test")
print("=" * 60)

from server.app import app

# Start server in background thread
server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=17860, log_level="warning"))
thread = threading.Thread(target=server.run, daemon=True)
thread.start()
time.sleep(3)  # Wait for startup

BASE = "http://127.0.0.1:17860"
try:
    # Health check
    r = requests.get(f"{BASE}/health", timeout=5)
    assert r.status_code == 200, f"Health failed: {r.status_code}"
    print(f"  /health: {r.json()}")

    # Schema check
    r = requests.get(f"{BASE}/schema", timeout=5)
    assert r.status_code == 200, f"Schema failed: {r.status_code}"
    keys = list(r.json().keys())
    print(f"  /schema: keys={keys}")

    # Reset
    r = requests.post(f"{BASE}/reset", json={"task": "smart_factory_easy"}, timeout=5)
    assert r.status_code == 200, f"Reset failed: {r.status_code}"
    data = r.json()
    obs = data["observation"]
    print(f"  /reset: pos={obs['robot_pos']} carrying={obs['carrying']} grid={len(obs['grid_layout'])}x{len(obs['grid_layout'][0])}")

    # Step
    r = requests.post(f"{BASE}/step", json={"action": {"action": 5}}, timeout=5)
    assert r.status_code == 200, f"Step failed: {r.status_code}"
    data = r.json()
    obs = data["observation"]
    print(f"  /step: pos={obs['robot_pos']} reward={data['reward']:.3f} done={data['done']}")

    # Run a full easy episode
    r = requests.post(f"{BASE}/reset", json={"task": "smart_factory_easy"}, timeout=5)
    data = r.json()
    obs = data["observation"]
    total_reward = 0
    for i in range(30):
        if obs["carrying"] == 0:
            y, x = obs["robot_pos"]
            if y > 0: act = 5
            elif x > 0: act = 3
            else: act = 1
        else:
            y, x = obs["robot_pos"]
            gs = len(obs["grid_layout"]) - 1
            if y < gs: act = 6
            elif x < gs: act = 4
            else: act = 2
        r = requests.post(f"{BASE}/step", json={"action": {"action": act}}, timeout=5)
        data = r.json()
        obs = data["observation"]
        total_reward += data.get("reward", 0)
        if data["done"]:
            break

    print(f"  Full easy episode: {i+1} steps, reward={total_reward:.3f}, deliveries={obs['deliveries_made']}/{obs['deliveries_required']}")

    print("\n✅ TEST 2 PASSED: HTTP server works correctly\n")

except Exception as e:
    print(f"\n❌ TEST 2 FAILED: {e}\n")
    import traceback
    traceback.print_exc()
finally:
    server.should_exit = True
    time.sleep(1)

print("=" * 60)
print("ALL TESTS COMPLETED")
print("=" * 60)
