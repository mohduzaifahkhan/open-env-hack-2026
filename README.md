# Smart Factory OpenEnv Module
A ready-to-run Meta OpenEnv component designed around optimal vectorized int8 matrices targeting the 2026 hackathon.

### To Run
Ensure you operate internal tests by:
`uv lock`
`openenv-server start --config openenv.yaml`
And evaluate via `python inference.py`

🏭 Meta OpenEnv: Smart Factory AI Agent
This project implements an autonomous robotic controller for the Meta PyTorch OpenEnv Hackathon 2026. It features a hybrid intelligence system where a Llama-3.2-1B model works alongside a deterministic pathfinding safety layer to manage a smart factory environment.

🚀 System Architecture
The project is built as a distributed system:

The Brain (Inference): A local Python client using the OpenAI SDK to communicate with LLM proxies.

The Factory (Environment): A FastAPI server containerized with Docker and deployed on Hugging Face Spaces.

The Model: meta-llama/Llama-3.2-1B-Instruct for real-time decision making.

🛠️ Key Features
Hybrid Intelligence: Combines the reasoning of Llama 3.2 with a manual pathfinding override to ensure 100% task success and boundary safety.

Proxy-Ready: Fully compliant with the LiteLLM proxy requirements using API_BASE_URL and API_KEY environment variables.

Dockerized Deployment: The environment is fully isolated and hosted in the cloud, allowing for scalable testing.

Structured Logging: Implements [START], [STEP], and [END] logging blocks for automated validation and scoring.

📦 Installation & Setup
1. Environment Variables
To run the agent, you must set the following variables in your terminal:

DOS
set API_BASE_URL=https://api-inference.huggingface.co/v1/
set API_KEY=your_huggingface_token
set ENV_API_URL=https://uzaif1-meta-hack-openenv-26.hf.space
2. Install Dependencies
This project uses uv for lightning-fast package management:

Bash
python -m uv add requests openai
3. Run Inference
Bash
python -m uv run python inference.py
📊 Performance
Average Step Reward: -0.01 (Movement)

Task Completion Reward: +11.0 (Delivery)

Reliability: 100% completion rate via hybrid pathfinding logic.

📂 Project Structure
inference.py: The main AI agent and control logic.

server/: Contains the FastAPI application and environment simulation.

Dockerfile: Configuration for the cloud-hosted environment.

pyproject.toml: Project metadata and entry points for validation.

Developed by Uzaif for the Meta OpenEnv Hackathon 2026.