from huggingface_hub import HfApi
import os

api = HfApi()
api.upload_folder(
    folder_path=".",
    repo_id="VarunKumar264/email-triage-openenv",
    repo_type="space",
    ignore_patterns=["upload.py", ".git"],
    token=os.environ.get("HF_TOKEN")
)
print("Upload successful!")