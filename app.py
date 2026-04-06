from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
from env.email_triage_env import EmailTriageEnv, Action, Observation
import uvicorn

app = FastAPI()
env = EmailTriageEnv()

@app.get("/")
def root():
    return {"status": "ok", "message": "Email Triage OpenEnv is running!"}

@app.post("/reset")
def reset():
    obs = env.reset()
    return obs.dict()

@app.post("/step")
def step(action: Action):
    obs, reward, done, info = env.step(action)
    return {"observation": obs.dict(), "reward": reward, "done": done, "info": info}

@app.get("/state")
def state():
    return env.state()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)