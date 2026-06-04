"""Push repo to a Hugging Face Space on every main-branch commit.

Required env vars (set as GitHub repository secrets):
  HF_TOKEN    — HF access token with write access to the target space
  HF_SPACE_ID — e.g. "your-username/vocal-coach"
"""

import os
import sys
from huggingface_hub import HfApi

HF_TOKEN = os.environ.get("HF_TOKEN", "")
HF_SPACE_ID = os.environ.get("HF_SPACE_ID", "")

if not HF_TOKEN or not HF_SPACE_ID:
    print("ERROR: HF_TOKEN and HF_SPACE_ID must be set as GitHub secrets.")
    sys.exit(1)

api = HfApi(token=HF_TOKEN)

# Create the space if it doesn't exist yet
api.create_repo(repo_id=HF_SPACE_ID, repo_type="space", exist_ok=True)
print(f"Space: https://huggingface.co/spaces/{HF_SPACE_ID}")

# Prepend HF Spaces YAML front-matter to the project README so HF renders it
# correctly as a Docker-based space.  The rest of the README is kept as-is.
with open("README.md") as f:
    original_readme = f.read()

hf_readme = (
    "---\n"
    "title: VocalCoach\n"
    "emoji: \U0001f3a4\n"
    "colorFrom: indigo\n"
    "colorTo: pink\n"
    "sdk: docker\n"
    "pinned: false\n"
    "---\n\n"
    + original_readme
)

with open("README.md", "w") as f:
    f.write(hf_readme)

# Upload the full repo.  huggingface_hub automatically routes files >10 MB
# through Git LFS (the .pth checkpoints are 30–90 MB each).
api.upload_folder(
    folder_path=".",
    repo_id=HF_SPACE_ID,
    repo_type="space",
    ignore_patterns=[
        "**/__pycache__/**",
        "**/*.pyc",
        "**/*.pyo",
        ".git/**",
        ".github/**",
        "vocalcoach_sessions/**",   # runtime-generated; not part of the demo image
    ],
)

print(f"Deploy complete: https://huggingface.co/spaces/{HF_SPACE_ID}")
