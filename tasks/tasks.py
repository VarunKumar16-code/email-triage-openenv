from env.email_triage_env import EmailTriageEnv, Action
from typing import Dict

# ---- Task 1: Easy — Single urgent email ----
def task_easy_grader(agent_fn) -> float:
    env = EmailTriageEnv(task_id=0, max_steps=3)
    obs = env.reset()
    total_reward = 0.0
    steps = 0
    done = False
    while not done:
        action = agent_fn(obs)
        obs, reward, done, info = env.step(action)
        total_reward += reward
        steps += 1
    score = round(total_reward / steps, 4)
    assert 0.0 <= score <= 1.0
    return score

# ---- Task 2: Medium — Mixed inbox ----
def task_medium_grader(agent_fn) -> float:
    env = EmailTriageEnv(task_id=1, max_steps=5)
    obs = env.reset()
    total_reward = 0.0
    steps = 0
    done = False
    while not done:
        action = agent_fn(obs)
        obs, reward, done, info = env.step(action)
        total_reward += reward
        steps += 1
    score = round(total_reward / steps, 4)
    assert 0.0 <= score <= 1.0
    return score

# ---- Task 3: Hard — Full inbox + reply requirement ----
def task_hard_grader(agent_fn) -> float:
    env = EmailTriageEnv(task_id=2, max_steps=8)
    obs = env.reset()
    total_reward = 0.0
    steps = 0
    done = False
    while not done:
        action = agent_fn(obs)
        # Hard task penalizes missing replies on urgent emails
        if hasattr(obs, 'subject') and 'URGENT' in obs.subject.upper() and (not action.reply or len(action.reply) < 10):
            obs, reward, done, info = env.step(action)
            total_reward += reward * 0.8  # 20% penalty for no reply
        else:
            obs, reward, done, info = env.step(action)
            total_reward += reward
        steps += 1
    score = round(min(total_reward / steps, 1.0), 4)
    assert 0.0 <= score <= 1.0
    return score

TASKS = {
    "easy": {"name": "Basic Email Triage", "grader": task_easy_grader, "difficulty": "easy"},
    "medium": {"name": "Mixed Inbox Triage", "grader": task_medium_grader, "difficulty": "medium"},
    "hard": {"name": "Full Inbox with Replies", "grader": task_hard_grader, "difficulty": "hard"},
}