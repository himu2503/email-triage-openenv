from environment import EmailTriageEnv
from models import Action

def test_easy_task():
    env = EmailTriageEnv("easy")
    obs = env.reset()

    action = Action(
        action_type="classify",
        label="urgent",
        priority="high",
        confidence=1.0
    )

    obs, reward, done, info = env.step(action)

    assert 0.0 <= reward.score <= 1.0