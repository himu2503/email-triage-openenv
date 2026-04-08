"""
OpenEnv Email Triage Environment — Typed Pydantic Models
Observation, Action, Reward
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any


class Email(BaseModel):
    """A single email item."""
    email_id: str
    subject: str
    body: str
    sender: str
    timestamp: str
    thread_history: List[Dict[str, str]] = Field(default_factory=list)


class Observation(BaseModel):
    """
    What the agent sees at each step.
    Contains the current email to process plus inbox context.
    """
    email_id: str
    subject: str
    body: str
    sender: str
    timestamp: str
    thread_history: List[Dict[str, str]] = Field(default_factory=list)
    inbox_size: int = 0
    step_number: int = 0
    task_id: str = ""
    task_description: str = ""

    model_config = {"json_schema_extra": {
        "example": {
            "email_id": "e1",
            "subject": "URGENT: Server down",
            "body": "Production is down. Please fix ASAP.",
            "sender": "ops@company.com",
            "timestamp": "2024-01-01T09:00:00",
            "inbox_size": 3,
            "step_number": 1,
            "task_id": "easy",
            "task_description": "Classify 3 clearly labeled emails"
        }
    }}


class Action(BaseModel):
    """
    What the agent does in response to an email.

    action_type options:
      - "classify"  → just assign a label (easy task)
      - "archive"   → mark as low priority / archive
      - "escalate"  → mark as urgent, escalate to human
      - "reply"     → draft a reply (hard task)
      - "delete"    → mark as spam/delete

    label options: urgent, spam, billing, hr, it_support,
                   inquiry, newsletter, internal, other
    """
    action_type: str = Field(
        ...,
        description="One of: classify, archive, escalate, reply, delete"
    )
    label: Optional[str] = Field(
        None,
        description="Category label: urgent, spam, billing, hr, it_support, inquiry, newsletter, internal, other"
    )
    reply_text: Optional[str] = Field(
        None,
        description="Draft reply text (required for action_type='reply')"
    )
    priority: Optional[str] = Field(
        None,
        description="Priority level: high, medium, low"
    )
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Agent confidence in this action (0.0–1.0)"
    )

    model_config = {"json_schema_extra": {
        "example": {
            "action_type": "escalate",
            "label": "urgent",
            "reply_text": None,
            "priority": "high",
            "confidence": 0.95
        }
    }}


class Reward(BaseModel):
    """
    Reward signal returned after each step.
    Always in range [0.0, 1.0].
    """
    score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Overall reward for this step (0.0–1.0)"
    )
    partial_credits: Dict[str, float] = Field(
        default_factory=dict,
        description="Breakdown of partial credit by component"
    )
    reason: str = Field(
        default="",
        description="Human-readable explanation of the reward"
    )
    correct_label: Optional[str] = Field(
        None,
        description="The expected label (revealed after grading)"
    )
    correct_action_type: Optional[str] = Field(
        None,
        description="The expected action type (revealed after grading)"
    )

    model_config = {"json_schema_extra": {
        "example": {
            "score": 0.8,
            "partial_credits": {"label": 0.5, "action_type": 0.3},
            "reason": "Correct label, wrong action type",
            "correct_label": "urgent",
            "correct_action_type": "escalate"
        }
    }}


class EpisodeResult(BaseModel):
    """Summary of a completed episode."""
    task_id: str
    total_steps: int
    total_reward: float
    avg_reward: float
    rewards_per_step: List[float]
    actions_taken: List[Dict[str, Any]]
