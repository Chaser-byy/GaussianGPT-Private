from huggingface_hub import HfApi
import os

repo_id = "GaussianWorld/aria_synthetic_envs_mcmc_3dgs_new"
api = HfApi(endpoint=os.environ["HF_ENDPOINT"])

# 先列顶层
top_items = list(api.list_repo_tree(
    repo_id=repo_id,
    repo_type="dataset",
    recursive=False,
    token=os.environ["HF_TOKEN"],
))

folders = [x.path for x in top_items if type(x).__name__ == "RepoFolder"]

print("First 20 folders:")
for f in folders[:20]:
    print(repr(f))

print("\nLast 20 folders:")
for f in folders[-20:]:
    print(repr(f))

scene_id = folders[0]
print("\nNow listing first folder:", repr(scene_id))

items = api.list_repo_tree(
    repo_id=repo_id,
    repo_type="dataset",
    path_in_repo=scene_id,
    recursive=True,
    token=os.environ["HF_TOKEN"],
)

for item in items:
    size = getattr(item, "size", None)
    print(item.path, type(item).__name__, size)