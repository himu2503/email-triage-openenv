import os
import time
from openai import OpenAI

from environment import EmailTriageEnv
from models import Action

# ─── ENV CONFIG ─────────────────────────────────────

API_BASE_URL = os.getenv("API_BASE_URL", "https://api-inference.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")
HF_TOKEN     = os.getenv("HF_TOKEN", "")

client = OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)

TASK_IDS = ["easy", "medium", "hard"]

SYSTEM_PROMPT = """You are an expert email triage assistant.

Respond ONLY with JSON:
{
  "action_type": "...",
  "label": "...",
  "priority": "...",
  "reply_text": "...",
  "confidence": 0.0
}
"""

# ─── HELPERS ───────────────────────────────────────

def build_prompt(obs):
    return f"""
Email #{obs.step_number}/{obs.inbox_size}

From: {obs.sender}
Subject: {obs.subject}

{obs.body}

Return JSON only.
"""

def call_llm(prompt):
    try:
        res = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            max_tokens=300,
        )
        return res.choices[0].message.content
    except:
        return '{"action_type":"classify","label":"other","priority":"medium","confidence":0.0}'

def parse_action(text):
    import json
    try:
        data = json.loads(text)
        return Action(**data)
    except:
        return Action(
            action_type="classify",
            label="other",
            priority="medium",
            confidence=0.0
        )

# ─── RUN TASK ──────────────────────────────────────

def run_task(task_id):
    env = EmailTriageEnv(task_id=task_id)
    obs = env.reset()

    total_reward = 0.0
    rewards = []
    step_num = 0

    # START
    print(f"[START] task={task_id} env=email-triage model={MODEL_NAME}", flush=True)

    while True:
        if obs.email_id == "__done__":
            break

        prompt = build_prompt(obs)
        raw = call_llm(prompt)
        action = parse_action(raw)

        obs, reward, done, info = env.step(action)

        step_num += 1
        r = reward.score
        total_reward += r
        rewards.append(r)

        # STEP
        print(
            f"[STEP] step={step_num} "
            f"action={action.action_type}:{action.label}:{action.priority} "
            f"reward={r:.2f} "
            f"done={str(done).lower()} "
            f"error=null",
            flush=True
        )

        if done:
            break

    avg = total_reward / step_num if step_num > 0 else 0.0
    success = avg >= 0.5

    rewards_str = ",".join(f"{x:.2f}" for x in rewards)

    # END
    print(
        f"[END] success={str(success).lower()} "
        f"steps={step_num} "
        f"score={avg:.2f} "
        f"rewards={rewards_str}",
        flush=True
    )

    return avg

# ─── MAIN ──────────────────────────────────────────

def main():
    results = []

    for t in TASK_IDS:
        avg = run_task(t)
        results.append(avg)
        time.sleep(1)

    overall = sum(results) / len(results)

    print(f"\nFinal Avg Score: {overall:.2f}")

if __name__ == "__main__":
    main()