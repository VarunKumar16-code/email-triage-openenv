import os
import json
import sys
from openai import OpenAI
from env.email_triage_env import EmailTriageEnv, Action
from tasks.tasks import TASKS

# ---------- Env config ----------
API_BASE_URL = os.getenv("API_BASE_URL", "https://api-inference.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3.2-3B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN", "")

if not HF_TOKEN:
    print(json.dumps({"type": "ERROR", "message": "HF_TOKEN not set"}))
    sys.exit(1)

client = OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)

SYSTEM_PROMPT = """You are an expert email triage assistant.
Given an email, classify it as exactly one of: urgent, normal, spam, delete.
If urgent, also write a brief professional reply.
Respond ONLY in JSON: {"label": "...", "reply": "...or null"}"""

def agent_fn(obs):
    prompt = f"""Subject: {obs.subject}
From: {obs.sender}
Body: {obs.body}
Classify this email."""
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            max_tokens=200,
        )
        raw = response.choices[0].message.content.strip()
    except Exception as e:
        print(json.dumps({"type": "WARN", "message": f"API call failed: {e}"}))
        raw = ""

    try:
        parsed = json.loads(raw)
    except Exception:
        parsed = {"label": "normal", "reply": None}

    label = parsed.get("label", "normal")
    if label not in ("urgent", "normal", "spam", "delete"):
        label = "normal"
    reply = parsed.get("reply", None)
    return Action(email_id=obs.email_id, label=label, reply=reply)

def run_task(task_key: str):
    try:
        task = TASKS[task_key]
        print(json.dumps({"type": "START", "task": task_key, "difficulty": task["difficulty"]}))
        env = EmailTriageEnv(max_steps={"easy": 3, "medium": 5, "hard": 8}[task_key])
        obs = env.reset()
        step = 0
        total_reward = 0.0
        done = False
        while not done:
            action = agent_fn(obs)
            obs, reward, done, info = env.step(action)
            total_reward += reward
            step += 1
            print(json.dumps({
                "type": "STEP",
                "step": step,
                "action_label": action.label,
                "reward": reward,
                "reason": info["reason"],
            }))
        final_score = round(total_reward / step, 4)
        print(json.dumps({
            "type": "END",
            "task": task_key,
            "total_steps": step,
            "final_score": final_score,
        }))
        return final_score
    except Exception as e:
        print(json.dumps({"type": "ERROR", "task": task_key, "message": str(e)}))
        return 0.0

if __name__ == "__main__":
    scores = {}
    for task_key in ["easy", "medium", "hard"]:
        scores[task_key] = run_task(task_key)
    print(json.dumps({"type": "FINAL", "all_scores": scores}))