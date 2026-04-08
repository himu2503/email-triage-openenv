# 📧 Email Triage OpenEnv

An [OpenEnv](https://github.com/huggingface/openenv)-compliant environment for training and evaluating AI agents on **real-world email triage**: classification, routing, prioritization, and reply drafting.

---

## 🌍 Environment Description

Email triage is a genuine daily task for knowledge workers — sorting, routing, and responding to emails efficiently and correctly. This environment presents an AI agent with a stream of emails and asks it to make decisions that reflect real business logic:

- Is this urgent or spam?
- Should this go to IT, HR, or billing?
- What priority does this warrant?
- Does this need a reply, and what should it say?

The environment provides dense reward signals (not just end-of-episode), enabling agents to learn from partial correctness.

---

## 🗂️ Action Space

| Field | Type | Values |
|---|---|---|
| `action_type` | categorical | `classify`, `archive`, `escalate`, `reply`, `delete` |
| `label` | categorical | `urgent`, `spam`, `billing`, `hr`, `it_support`, `inquiry`, `newsletter`, `internal`, `other` |
| `priority` | categorical | `high`, `medium`, `low` |
| `reply_text` | string (optional) | Free-text reply draft |
| `confidence` | float [0,1] | Agent confidence |

---

## 👁️ Observation Space

| Field | Type | Description |
|---|---|---|
| `email_id` | string | Unique email identifier |
| `subject` | string | Email subject line |
| `body` | string | Email body text |
| `sender` | string | Sender email address |
| `timestamp` | string | ISO 8601 timestamp |
| `thread_history` | list | Prior messages in thread |
| `inbox_size` | int | Total emails in this task |
| `step_number` | int | Current step (1-indexed) |
| `task_id` | string | Current task identifier |
| `task_description` | string | Human-readable task description |

---

## 🎯 Tasks

### Task 1: Easy — `easy`
**3 emails** | Classify only | Expected score: ~0.80

Three clearly labeled emails: a production outage (urgent), a prize scam (spam), and a vendor invoice (billing). The agent just needs to assign the correct label.

| Reward Component | Weight |
|---|---|
| Correct label | 100% |

---

### Task 2: Medium — `medium`
**5 emails** | Classify + Route + Prioritize | Expected score: ~0.55

Five more ambiguous emails: emergency leave request, IT helpdesk issue, internal project update, newsletter, overdue invoice. The agent must classify, choose the right action type (escalate vs archive vs classify), and set the right priority.

| Reward Component | Weight |
|---|---|
| Correct label | 50% |
| Correct action type | 30% |
| Correct priority | 20% |

---

### Task 3: Hard — `hard`
**8 emails** | Full pipeline + Reply drafting | Expected score: ~0.40

Eight complex, realistic emails including a data breach alert, a borderline refund request, a partnership inquiry, and a payroll complaint. The agent must classify, route, prioritize, AND draft appropriate replies where needed. Reply quality is scored by keyword matching against expected content.

| Reward Component | Weight |
|---|---|
| Correct label | 30% |
| Correct action type | 25% |
| Correct priority | 15% |
| Reply quality | 30% |

---

## 🏆 Reward Function

Rewards are dense — the agent receives signal at every step, not just at episode end.

**Reward shaping:**
- ✅ Partial credit for getting some components right (label only, or label + action_type)
- ✅ +0.05 bonus when agent confidence ≥ 0.9 and score ≥ 0.8
- ❌ -0.1 penalty for missing label (null/empty)
- ❌ -0.05 penalty for replies > 1000 chars (verbose padding)

All rewards clipped to [0.0, 1.0].

---

## 🚀 Setup & Usage

### Option 1: Docker (recommended)

```bash
git clone https://huggingface.co/spaces/<your-username>/email-triage-env
cd email-triage-env

docker build -t email-triage-env .
docker run -p 7860:7860 \
  -e API_BASE_URL="https://api-inference.huggingface.co/v1" \
  -e MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct" \
  -e HF_TOKEN="your_hf_token" \
  email-triage-env
```

### Option 2: Local Python

```bash
pip install -r requirements.txt
uvicorn server:app --host 0.0.0.0 --port 7860
```

---

## 🔌 API Usage

### Reset environment
```bash
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "easy"}'
```

### Take a step
```bash
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{
    "action_type": "escalate",
    "label": "urgent",
    "priority": "high",
    "confidence": 0.95
  }'
```

### Get environment state
```bash
curl http://localhost:7860/state
```

---

## 🤖 Running the Baseline

```bash
export API_BASE_URL="https://api-inference.huggingface.co/v1"
export MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
export HF_TOKEN="your_token_here"

python inference.py
```

**Expected baseline output (approximate):**

```json
{"type": "[END]", "task_id": "easy",   "avg_reward": 0.85}
{"type": "[END]", "task_id": "medium", "avg_reward": 0.52}
{"type": "[END]", "task_id": "hard",   "avg_reward": 0.38}
{"type": "SUMMARY", "overall_avg": 0.58}
```

---

## 📁 Project Structure

```
email-triage-env/
├── models.py        ← Pydantic Observation / Action / Reward
├── tasks.py         ← Email datasets + 3 graders
├── environment.py   ← EmailTriageEnv (step/reset/state)
├── server.py        ← FastAPI REST server
├── inference.py     ← Baseline inference script (root level)
├── openenv.yaml     ← OpenEnv metadata
├── Dockerfile       ← Container config
├── requirements.txt
└── README.md
```

---

## 📊 Why Email Triage?

- **Universal**: Every organization handles email — this is a real, daily task
- **Measurable**: Classification and routing have clear right/wrong answers
- **Scalable**: Difficulty scales naturally (classify → route → draft reply)
- **Useful for RL**: Dense reward signal, partial progress, realistic action space
- **Frontier-model challenging**: The hard task requires reasoning, tone awareness, and policy judgment that trips up even capable models

---

## 🏷️ Tags

`openenv` `email` `triage` `nlp` `classification` `real-world` `business`
