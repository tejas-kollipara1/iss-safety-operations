---
title: ISS Safety Operations
emoji: 🚀
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
---

# ISS Safety Operations Environment (OpenEnv)

## Description
This environment simulates a real-world task where an AI agent acts as a Mission Control Safety Officer aboard the International Space Station (ISS). The environment supports multi-mode safety operations across three distinct episodes:
1. **`audit_001`** (Easy): Pre-EVA safety audit, find expired equipment.
2. **`emergency_001`** (Medium): Live fire in Lab Module, protect crew and contain.
3. **`investigation_001`** (Hard): Post-anomaly root cause investigation with incomplete logs.

The goal is to safely resolve these situations using the physical safety objects across the station modules. The agent earns rewards for making correct safety and investigation decisions.

## Action & Observation Spaces

### Observation Space
**Type:** `structured_json`
The observation provides the agent with structured JSON detailing situational awareness of the ISS modules, active alarms, available safety equipment, and crew status.

### Action Space
**Type:** `categorical_with_target`
The agent can select discrete JSON actions directed at specific modules or target objects. Required JSON schema:
```json
{
  "action_type": "<action_name>",
  "target_object_id": "<object_id>",
  "target_module": "<module_name>",
  "reasoning": "<short justification>"
}
```

The valid actions include:
- `inspect_object`, `flag_non_compliant`, `submit_report`
- `evacuate_module`, `deploy_resource`, `trigger_switch`, `contact_ground`
- `pull_sensor_log`, `cross_reference`, `identify_root_cause`

## Setup Instructions

1. Ensure you have the `openenv-core` framework and dependencies installed:
```bash
pip install -r requirements.txt
```

2. Standard configuration parameters must be set as environment variables before executing the baseline inference:
```bash
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="meta-llama/Llama-3-8b-instruct:groq"
export HF_TOKEN="your_hf_token_here"
```

3. Run the automated baseline test script:
```bash
python inference.py
```
