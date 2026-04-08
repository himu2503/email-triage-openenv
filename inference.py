"""
OpenEnv Email Triage — Baseline Inference Script

Runs an LLM agent against all 3 tasks (easy, medium, hard) using the OpenAI client.
Emits structured logs in the required [START] / [STEP] / [END] format.

Required environment variables:
  API_BASE_URL  — LLM API base URL
  MODEL_NAME    — model identifier
  HF_TOKEN      — API key / HuggingFace token

Usage:
  python inference.py
"""

import os
import sys
import json
import time
from typing import Optional

from openai import OpenAI
from environment import EmailTriageEnv
from models import Action


# ─── Config ────────────────────────────────────────────

API_BASE_URL = os.environ.get("API_BASE_URL", "https://api-inference.huggingface.co/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME",   "meta-llama/Llama-3.1-8B-Instruct")
HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("OPENAI_API_KEY")

if not HF_TOKEN:
    print("[WARNING] HF_TOKEN not set — API calls will likely fail.", file=sys.stderr)

client = OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)

TASK_IDS = ["easy", "medium", "hard"]

# System prompt given to the agent
SYSTEM_PROMPT = """You are an expert email triage assistant. 
For each email you receive, you must respond with a JSON object (and nothing else).

JSON fields:
- action_type (required): one of "classify", "archive", "escalate", "reply", "delete"
- label (required): one of "urgent", "spam", "billing", "hr", "it_support", "inquiry", "newsletter", "internal", "other"
- priority (required): one of "high", "medium", "low"
- reply_text (optional): a professional reply if action_type is "reply" or "escalate"
- confidence: a float between 0.0 and 1.0

Rules:
- urgent/security/outage emails → escalate, high priority
- spam/promotional → delete or archive, low priority
- billing/invoices → classify or escalate, medium-high priority
- HR/payroll → escalate, high priority
- IT issues → escalate if urgent, medium priority
- Internal updates → classify or archive, medium-low priority
- Customer inquiries → reply with a helpful, professional response
- Partnership/sales outreach → reply politely

Always respond with ONLY valid JSON. No markdown, no explanation, no backticks."""


def build_prompt(obs) -> str:
    """Convert observation into a prompt string."""
    thread_ctx = ""
    if obs.thread_history:
        thread_ctx = "\n\nThread history:\n" + "\n".join(
            f"  [{m.get('role', 'unknown')}]: {m.get('content', '')[:200]}"
            for m in obs.thread_history[-3:]
        )

    return (
        f"Email #{obs.step_number} of {obs.inbox_size}\n"
        f"Task: {obs.task_description}\n\n"
        f"From: {obs.sender}\n"
        f"Subject: {obs.subject}\n"
        f"Time: {obs.timestamp}\n\n"
        f"Body:\n{obs.body}"
        f"{thread_ctx}\n\n"
        "Respond with JSON only:"
    )


def parse_action(raw: str) -> Action:
    """Parse LLM output into an Action. Handles common formatting issues."""
    # Strip markdown code blocks if present
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        cleaned = "\n".join(
            line for line in lines
            if not line.strip().startswith("```")
        ).strip()

    try:
        data = json.loads(cleaned)
        # Normalise label to lowercase
        if "label" in data and data["label"]:
            data["label"] = str(data["label"]).lower().strip()
        if "action_type" in data and data["action_type"]:
            data["action_type"] = str(data["action_type"]).lower().strip()
        if "priority" in data and data["priority"]:
            data["priority"] = str(data["priority"]).lower().strip()
        return Action(**data)
    except Exception:
        # Fallback: default safe action
        return Action(action_type="classify", label="other", priority="medium", confidence=0.1)


def call_llm(prompt: str, max_retries: int = 3) -> str:
    """Call the LLM with retry logic."""
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
                max_tokens=400,
                temperature=0.1,
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"[ERROR] LLM call failed after {max_retries} attempts: {e}", file=sys.stderr)
                return '{"action_type": "classify", "label": "other", "priority": "medium", "confidence": 0.0}'
            time.sleep(2 ** attempt)
    return ""


def run_task(task_id: str) -> float:
    """Run one full episode on a task. Returns average reward."""
    env = EmailTriageEnv(task_id=task_id)
    obs = env.reset()

    total_reward = 0.0
    step_num = 0
    rewards = []

    # ── [START] log ──────────────────────────────
    print(json.dumps({
        "type": "[START]",
        "task_id": task_id,
        "model": MODEL_NAME,
        "total_emails": obs.inbox_size,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }), flush=True)

    while True:
        if obs.email_id == "__done__":
            break

        prompt = build_prompt(obs)
        raw_output = call_llm(prompt)
        action = parse_action(raw_output)

        obs, reward, done, info = env.step(action)

        step_num += 1
        total_reward += reward.score
        rewards.append(reward.score)

        # ── [STEP] log ───────────────────────────
        print(json.dumps({
            "type": "[STEP]",
            "task_id": task_id,
            "step": step_num,
            "email_id": info.get("email_id", ""),
            "action": {
                "action_type": action.action_type,
                "label": action.label,
                "priority": action.priority,
                "confidence": action.confidence,
                "has_reply": bool(action.reply_text),
            },
            "reward": reward.score,
            "partial_credits": reward.partial_credits,
            "correct_label": reward.correct_label,
            "cumulative_reward": info.get("cumulative_reward", 0),
            "done": done,
        }), flush=True)

        if done:
            break

    avg_reward = round(total_reward / step_num, 4) if step_num > 0 else 0.0

    # ── [END] log ────────────────────────────────
    print(json.dumps({
        "type": "[END]",
        "task_id": task_id,
        "total_steps": step_num,
        "total_reward": round(total_reward, 4),
        "avg_reward": avg_reward,
        "rewards_per_step": rewards,
        "model": MODEL_NAME,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }), flush=True)

    return avg_reward


def main():
    print(json.dumps({
        "type": "INFO",
        "message": "Starting Email Triage OpenEnv baseline inference",
        "model": MODEL_NAME,
        "api_base": API_BASE_URL,
        "tasks": TASK_IDS,
    }), flush=True)

    results = {}
    for task_id in TASK_IDS:
        avg = run_task(task_id)
        results[task_id] = avg
        time.sleep(1)  # Brief pause between tasks

    print(json.dumps({
        "type": "SUMMARY",
        "results": results,
        "overall_avg": round(sum(results.values()) / len(results), 4),
        "model": MODEL_NAME,
    }), flush=True)


if __name__ == "__main__":
    main()
