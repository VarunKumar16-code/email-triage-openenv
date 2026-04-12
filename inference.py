import json

# ================= RUN TASK =================
def run_task(task_key: str):
    print("[START]")

    # simulate some steps (no env / no LLM)
    for _ in range(3):
        print("[STEP]")

    print("[END]")
    print(json.dumps({"score": 0.7}))  # strictly between (0,1)

# ================= MAIN =================
if __name__ == "__main__":
    for task_key in ["easy", "medium", "hard"]:
        run_task(task_key)
