"""
OpenEnv Email Triage — FastAPI Server

Exposes the environment over HTTP with these endpoints:
  POST /reset          → start episode, get first observation
  POST /step           → take action, get (obs, reward, done, info)
  GET  /state          → get current environment state
  GET  /tasks          → list available tasks
  GET  /health         → health check
  GET  /               → environment info
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any

from models import Observation, Action, Reward
from environment import EmailTriageEnv


app = FastAPI(
    title="Email Triage OpenEnv",
    description=(
        "An OpenEnv-compliant environment for training and evaluating AI agents "
        "on email triage: classification, routing, prioritization, and reply drafting."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Shared environment instance (single-session server)
_env: Optional[EmailTriageEnv] = None


# ─── Request/Response models ───────────────────────────

class ResetRequest(BaseModel):
    task_id: str = "easy"


class StepResponse(BaseModel):
    observation: Observation
    reward: Reward
    done: bool
    info: Dict[str, Any]


class TaskInfo(BaseModel):
    id: str
    description: str
    difficulty: str
    total_steps: int


# ─── Endpoints ─────────────────────────────────────────

@app.get("/")
def root():
    return {
        "name": "Email Triage OpenEnv",
        "version": "1.0.0",
        "description": "OpenEnv environment for email triage agent evaluation",
        "tasks": ["easy", "medium", "hard"],
        "endpoints": {
            "reset": "POST /reset",
            "step":  "POST /step",
            "state": "GET  /state",
            "tasks": "GET  /tasks",
        },
        "openenv_spec": "https://github.com/huggingface/openenv",
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "env_initialized": _env is not None,
        "current_task": _env.task_id if _env else None,
    }


@app.get("/tasks")
def list_tasks():
    from tasks import TASKS
    return [
        TaskInfo(
            id=t["id"],
            description=t["description"],
            difficulty=t["difficulty"],
            total_steps=t["total_steps"],
        )
        for t in TASKS.values()
    ]


from typing import Optional

@app.post("/reset", response_model=Observation)
def reset(request: Optional[ResetRequest] = None):
    global _env
    try:
        task_id = request.task_id if request else "easy"
        _env = EmailTriageEnv(task_id=task_id)
        obs = _env.reset()
        return obs
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/step", response_model=StepResponse)
def step(action: Action):
    global _env
    if _env is None:
        raise HTTPException(
            status_code=400,
            detail="Environment not initialized. Call POST /reset first."
        )
    try:
        obs, reward, done, info = _env.step(action)
        return StepResponse(observation=obs, reward=reward, done=done, info=info)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/state")
def state():
    if _env is None:
        raise HTTPException(
            status_code=400,
            detail="Environment not initialized. Call POST /reset first."
        )
    return _env.state()


@app.get("/episode_result")
def episode_result():
    if _env is None:
        raise HTTPException(status_code=400, detail="Environment not initialized.")
    result = _env.get_episode_result()
    if result is None:
        raise HTTPException(status_code=400, detail="Episode not complete yet.")
    return result
