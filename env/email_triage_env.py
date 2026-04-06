from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import random
import json

# ---------- Typed Models ----------

class Observation(BaseModel):
    email_id: str
    subject: str
    body: str
    sender: str
    inbox_size: int
    current_step: int
    max_steps: int

class Action(BaseModel):
    email_id: str
    label: str          # "urgent", "normal", "spam", "delete"
    reply: Optional[str] = None

class Reward(BaseModel):
    value: float
    reason: str

# ---------- Email Pool ----------

EMAIL_POOL = [
    {"id": "e001", "subject": "URGENT: Server down", "body": "Production server is down. Immediate action needed.", "sender": "ops@company.com", "true_label": "urgent"},
    {"id": "e002", "subject": "Team lunch tomorrow", "body": "Hey, we're doing lunch at noon. Join us!", "sender": "john@company.com", "true_label": "normal"},
    {"id": "e003", "subject": "You won $1,000,000!!!", "body": "Click here to claim your prize now!", "sender": "noreply@scam.biz", "true_label": "spam"},
    {"id": "e004", "subject": "Q3 Report Due", "body": "Reminder: Q3 financial report is due by EOD Friday.", "sender": "finance@company.com", "true_label": "urgent"},
    {"id": "e005", "subject": "Newsletter: Weekly digest", "body": "Here are this week's top stories...", "sender": "digest@news.com", "true_label": "spam"},
    {"id": "e006", "subject": "Re: Project update", "body": "Thanks for the update. Let's sync on Thursday.", "sender": "alice@partner.com", "true_label": "normal"},
    {"id": "e007", "subject": "CRITICAL: Security breach detected", "body": "Unauthorized access to your account from IP 203.x.x.x.", "sender": "security@company.com", "true_label": "urgent"},
    {"id": "e008", "subject": "Buy cheap meds online", "body": "Best prices on prescription drugs, no prescription needed.", "sender": "pharma@spam.net", "true_label": "spam"},
    {"id": "e009", "subject": "Meeting rescheduled", "body": "The 3pm meeting has been moved to 4pm.", "sender": "bob@company.com", "true_label": "normal"},
    {"id": "e010", "subject": "Invoice #4421 overdue", "body": "Your invoice is 30 days overdue. Please pay immediately.", "sender": "billing@vendor.com", "true_label": "urgent"},
]

# ---------- Environment ----------

class EmailTriageEnv:
    def __init__(self, task_id: int = 0, max_steps: int = 5):
        self.task_id = task_id
        self.max_steps = max_steps
        self._state: Dict[str, Any] = {}
        self.reset()

    def reset(self) -> Observation:
        self._state = {
            "current_step": 0,
            "score": 0.0,
            "inbox": random.sample(EMAIL_POOL, min(self.max_steps, len(EMAIL_POOL))),
            "history": [],
            "done": False,
        }
        return self._get_observation()

    def step(self, action: Action) -> tuple[Observation, float, bool, Dict]:
        if self._state["done"]:
            raise RuntimeError("Episode is done. Call reset().")

        current_email = self._state["inbox"][self._state["current_step"]]
        reward_value, reason = self._compute_reward(action, current_email)

        self._state["score"] += reward_value
        self._state["history"].append({
            "email_id": action.email_id,
            "predicted": action.label,
            "true": current_email["true_label"],
            "reward": reward_value,
        })
        self._state["current_step"] += 1
        done = self._state["current_step"] >= self.max_steps
        self._state["done"] = done

        obs = self._get_observation() if not done else Observation(
            email_id="done", subject="", body="", sender="",
            inbox_size=0, current_step=self._state["current_step"], max_steps=self.max_steps
        )
        return obs, reward_value, done, {"reason": reason, "history": self._state["history"]}

    def state(self) -> Dict[str, Any]:
        return {
            "current_step": self._state["current_step"],
            "max_steps": self.max_steps,
            "score": self._state["score"],
            "history": self._state["history"],
            "done": self._state["done"],
        }

    def _get_observation(self) -> Observation:
        if self._state["current_step"] >= len(self._state["inbox"]):
            return Observation(email_id="done", subject="", body="", sender="",
                               inbox_size=0, current_step=self._state["current_step"], max_steps=self.max_steps)
        email = self._state["inbox"][self._state["current_step"]]
        return Observation(
            email_id=email["id"],
            subject=email["subject"],
            body=email["body"],
            sender=email["sender"],
            inbox_size=len(self._state["inbox"]) - self._state["current_step"],
            current_step=self._state["current_step"],
            max_steps=self.max_steps,
        )

    def _compute_reward(self, action: Action, email: Dict) -> tuple[float, str]:
        correct = action.label == email["true_label"]
        # Partial rewards
        if correct:
            base = 1.0
            # Bonus for replying to urgent emails
            if email["true_label"] == "urgent" and action.reply and len(action.reply) > 10:
                base = min(1.0, base + 0.0)  # already max
                reason = "Correct label + meaningful reply to urgent email."
            else:
                reason = f"Correct label: {action.label}."
        else:
            # Partial credit: penalize dangerous misclassifications more
            if email["true_label"] == "urgent" and action.label in ("spam", "delete"):
                base = 0.0
                reason = f"Critical miss: urgent email labeled as {action.label}."
            elif email["true_label"] == "spam" and action.label == "urgent":
                base = 0.1
                reason = "Spam labeled as urgent — minor penalty."
            else:
                base = 0.3
                reason = f"Wrong label: predicted {action.label}, true {email['true_label']}."
        return round(base, 2), reason