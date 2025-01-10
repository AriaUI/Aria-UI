# 🚀 Welcome to the M3A Agent Powered by Aria-UI for AndroidWorld!

We’re thrilled to release the **M3A Agent powered by Aria-UI**! 🎉 This integration enhances task success rates and brings seamless grounding instruction understanding to **AndroidWorld**. Follow the steps below to get started. 🚀

## 🛠️ How to Use M3A Agent with AndroidWorld

### 1️⃣ Clone AndroidWorld and Set Up the Environment

First, clone the latest version of **AndroidWorld** and install the required dependencies:

```bash
git clone https://github.com/google-research/android_world.git
cd android_world
```
For more details, check out the official [AndroidWorld GitHub repository](https://github.com/google-research/android_world?tab=readme-ov-file).

### 2️⃣ Merge M3A Agent Files with AndroidWorld

We’ve organized our files to match AndroidWorld’s directory structure. To integrate:

- Merge the provided `agents/` and `env/` directories with the respective directories in AndroidWorld.
- Place `run_aria_ui.py` in the root directory of AndroidWorld.

After merging, your directory structure should look like this:
```
AndroidWorld/
├── agents/
│   └── aria_ui_utils.py
│   └── m3a_aria_ui.py
├── env/
│   └── json_action.py
├── run_aria_ui.py
...
```

### 3️⃣ Deploy Aria-UI API 🌐
Under AndroidWorld/agents/aria_ui_utils.py, you’ll find how we connect to Aria-UI using an API. Follow these steps to set it up:
- Deploy Aria-UI with vLLM to serve an OpenAI-style API.
Reference the [vLLM OpenAI-Compatible Server](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html).
- Update the API key and base URL in `aria_ui_utils.py` to match your deployment. Replace the placeholders with your values:
```python
ariaui_api_key = "ariaui_api_key"
ariaui_api_base = "ariaui_api_base"
```

**Note**: You may also set up a simple API server with the vLLM inference script and [FastAPI](https://github.com/fastapi/fastapi).
### 4️⃣ Run the M3A Agent 🎉
Once everything is set up, run the agent with the following command:
```python
python run_aria_ui.py
```
Sit back and watch **Aria-UI** in action! 🚀

### 💡 Additional Notes
For troubleshooting or further customization, explore `aria_ui_utils.py` and `m3a_aria_ui.py` to understand how Aria-UI is integrated.

Enjoy using the M3A Agent powered by **Aria-UI**! 🎉