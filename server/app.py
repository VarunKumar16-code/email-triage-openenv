from fastapi import FastAPI
from env.email_triage_env import EmailTriageEnv, Action
import uvicorn

app = FastAPI()
env = EmailTriageEnv()

@app.get("/")
def root():
    return {"status": "ok"}

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

def main():
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()