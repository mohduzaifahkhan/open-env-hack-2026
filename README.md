# 🏭 TRIBUNAL: Smart Factory Environment (OpenEnv)

A fully compliant, multi-difficulty autonomous factory logistics environment built for the Meta PyTorch OpenEnv Hackathon.

## 🌟 Architecture Overview
This submission utilizes a robust Client-Server architecture:
* **The Environment (Server):** Hosted live on Hugging Face Spaces (`app.py`). It manages grid state, movement logic, and scoring.
* **The Agent (Client):** Run locally (`inference.py`). It uses Meta's Llama-3.2-1B-Instruct model with dynamic Few-Shot Decision Tree prompting to solve the environment.

## 📊 Environment Details
* **Observation Space:** * `robot_pos`: Current [y, x] coordinates.
  * `carrying`: Integer boolean (0 = empty, 1 = carrying part).
  * `grid_max`: Dynamic grid boundary indicating difficulty level.
* **Action Space:** `[0]=NOOP, [1]=GRAB, [2]=PLACE, [3]=LEFT, [4]=RIGHT, [5]=UP, [6]=DOWN`
* **Difficulties (Tasks):**
  1. `smart_factory_easy`: 5x5 Grid
  2. `smart_factory_medium`: 7x7 Grid
  3. `smart_factory_hard`: 10x10 Grid

## 🚀 How to Run the Inference Script (For Judges)

The environment is already running live on Hugging Face at `https://uzaif1-meta-hack-openenv-26.hf.space`. You do not need to boot the server.

To run the AI agent against the live environment, follow these steps on your local machine:

**1. Clone the repository:**
```bash
git clone [https://github.com/mohduzaifahkhan/open-env-hack-2026.git](https://github.com/mohduzaifahkhan/open-env-hack-2026.git)
cd open-env-hack-2026



2. Install dependencies:

Bash
pip install requests openai


3. Set your Environment Variables (Required):

You must provide your own Hugging Face Router API key to query the Llama model.

For Mac/Linux:
Bash
export API_BASE_URL="[https://router.huggingface.co/v1/](https://router.huggingface.co/v1/)"
export API_KEY="your_hf_token_here"

For Windows (Command Prompt):
DOS
set API_BASE_URL=[https://router.huggingface.co/v1/](https://router.huggingface.co/v1/)
set API_KEY=your_hf_token_here
4. Run the Agent:

Bash
python inference.py
The script will automatically execute all 3 difficulty tiers sequentially, dynamically adapting the LLM prompt to the changing grid sizes, and output the normalized scores.


### The Final Move
If you push your updated `inference.py`, `app.py`, and this `README.md` to GitHub, your submission is bulletproof. The judges will clone it, paste those exact commands, and watch your AI flawlessly solve the grid on their own monitors. 

Everything is in place. Are all your files successfully pushed to GitHub?