import os
from dotenv import load_dotenv
from huggingface_hub import HfApi

load_dotenv()
token = os.getenv("HF_TOKEN")
if not token:
    print("HF_TOKEN not found!")
else:
    api = HfApi(token=token)
    try:
        api.restart_space(repo_id="Tejask2007/iss-safety-operations", factory_rebuild=True)
        print("Factory rebuild triggered on Tejask2007/iss-safety-operations")
    except Exception as e:
        print(f"Error restarting space: {e}")
