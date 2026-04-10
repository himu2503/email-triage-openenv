"""
OpenEnv Email Triage Environment

Implements the full OpenEnv interface:
  - reset()  → initial Observation
  - step()   → (Observation, Reward, done, info)
  - state()  → current state dict

Domain: Email triage — the agent reads emails one by one and must
classify, prioritize, route, and (for hard task) draft replies.
"""

from typing import Tuple, Dict, Any, Optional
from models import Observation, Action, Reward, EpisodeResult
from tasks import TASKS


class EmailTriageEnv:
    """
    Email Triage Environment.

    The agent processes emails one at a time.
    Each step: agent receives one email (Observation), takes an Action,
    gets a Reward, and receives the next email.
    Episode ends when all emails in the task are processed.

    Reward shaping:
      - Correct classification always rewarded
      - Partial credit for getting some components right
      - Penalty applied for excessively long/irrelevant replies (hard task)
      - No reward for skipped/empty actions
    """

    VALID_TASK_IDS = ("easy", "medium", "hard")

    def __init__(self, task_id: str = "easy"):
        if task_id not in self.VALID_TASK_IDS:
            raise ValueError(f"task_id must be one of {self.VALID_TASK_IDS}, got '{task_id}'")
        self.task_id = task_id
        self.task = TASKS[task_id]
        self._step_idx = 0
        self._history: list = []
        self._rewards: list = []
        self._done = False

    # ──────────────────────────────────────────
    # Core OpenEnv interface
    # ──────────────────────────────────────────

    def reset(self) -> Observation:
        """
        Reset the environment to initial state.
        Returns the first email as an Observation.
        """
        self._step_idx = 0
        self._history = []
        self._rewards = []
        self._done = False
        return self._build_observation()

    def step(self, action: Action) -> Tuple[Observation, Reward, bool, Dict[str, Any]]:
        """
        Process one action (agent's response to current email).

        Args:
            action: Agent's Action (label, action_type, priority, reply_text)

        Returns:
            observation: Next email (or terminal obs if done)
            reward:      Reward for this step
            done:        True if episode is complete
            info:        Metadata dict
        """
        if self._done:
            raise RuntimeError("Episode is done. Call reset() to start a new episode.")

        current_email = self._current_email()

        # Apply reward shaping penalty for overly long replies (spam prevention)
        reward = self.task["grader"](current_email, action, self._history)
        reward = self._apply_reward_shaping(action, reward)

        # Record history
        self._history.append({
            "email_id": current_email["email_id"],
            "action": action.model_dump(),
            "reward": reward.score,
        })
        self._rewards.append(reward.score)

        # Advance step
        self._step_idx += 1
        self._done = self._step_idx >= len(self.task["emails"])

        # Build next observation
        if self._done:
            next_obs = self._terminal_observation()
        else:
            next_obs = self._build_observation()

        info = {
            "task_id": self.task_id,
            "step": self._step_idx,
            "total_steps": len(self.task["emails"]),
            "cumulative_reward": round(sum(self._rewards), 3),
            "avg_reward_so_far": round(sum(self._rewards) / len(self._rewards), 3),
            "email_id": current_email["email_id"],
        }

        return next_obs, reward, self._done, info

    def state(self) -> Dict[str, Any]:
        """
        Returns current state of the environment (without email content).
        """
        return {
            "task_id": self.task_id,
            "task_description": self.task["description"],
            "difficulty": self.task["difficulty"],
            "current_step": self._step_idx,
            "total_steps": len(self.task["emails"]),
            "done": self._done,
            "cumulative_reward": round(sum(self._rewards), 3) if self._rewards else 0.0,
            "avg_reward": round(sum(self._rewards) / len(self._rewards), 3) if self._rewards else 0.0,
            "history_length": len(self._history),
        }

    # ──────────────────────────────────────────
    # Helper methods
    # ──────────────────────────────────────────

    def _current_email(self) -> Dict:
        return self.task["emails"][self._step_idx]

    def _build_observation(self) -> Observation:
        email = self._current_email()
        return Observation(
            **email,
            inbox_size=len(self.task["emails"]),
            step_number=self._step_idx + 1,
            task_id=self.task_id,
            task_description=self.task["description"],
        )

    def _terminal_observation(self) -> Observation:
        """Returned when episode is complete."""
        return Observation(
            email_id="__done__",
            subject="Episode Complete",
            body=f"All {len(self.task['emails'])} emails processed. Final avg reward: {round(sum(self._rewards)/len(self._rewards), 3) if self._rewards else 0}",
            sender="system",
            timestamp="",
            inbox_size=0,
            step_number=self._step_idx,
            task_id=self.task_id,
            task_description=self.task["description"],
        )

    def _apply_reward_shaping(self, action: Action, reward: Reward) -> Reward:
        """
        Apply additional reward shaping rules:
        - Penalize empty/null label for non-spam emails (-0.1)
        - Penalize suspiciously long replies (>1000 chars) with a small deduction
        - Bonus for high confidence that turns out correct
        """
        score = reward.score
        partial = dict(reward.partial_credits)

        # Penalize missing label
        if not action.label:
            penalty = -0.1
            score = max(0.01, score + penalty)
            partial["missing_label_penalty"] = penalty

        # Penalize excessively long replies (>1000 chars suggests padding)
        if action.reply_text and len(action.reply_text) > 1000:
            penalty = -0.05
            score = max(0.01, score + penalty)
            partial["verbose_reply_penalty"] = penalty

        # Small bonus if agent has high confidence AND is correct
        if action.confidence >= 0.9 and reward.score >= 0.8:
            bonus = 0.05
            score = min(0.99, score + bonus)
            partial["high_confidence_correct_bonus"] = bonus

        return Reward(
            score=round(score, 3),
            partial_credits=partial,
            reason=reward.reason,
            correct_label=reward.correct_label,
            correct_action_type=reward.correct_action_type,
        )

    def get_episode_result(self) -> Optional[EpisodeResult]:
        """Returns episode summary (only available after episode is done)."""
        if not self._done:
            return None
        return EpisodeResult(
            task_id=self.task_id,
            total_steps=len(self._rewards),
            total_reward=round(sum(self._rewards), 3),
            avg_reward=round(sum(self._rewards) / len(self._rewards), 3),
            rewards_per_step=self._rewards,
            actions_taken=self._history,
        )