from huggingface_hub import HfApi
import os

repo_id = "GaussianWorld/aria_synthetic_envs_mcmc_3dgs_new"

api = HfApi()

items = api.list_repo_tree(
    repo_id=repo_id,
    repo_type="dataset",
    recursive=False,
    token=os.environ["HF_TOKEN"],
)

print("Top-level files/folders:", flush=True)

for item in items:
    print(item.path, type(item).__name__, flush=True)