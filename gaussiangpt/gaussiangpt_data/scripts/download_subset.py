from huggingface_hub import hf_hub_download
from pathlib import Path
import os

repo_id = "GaussianWorld/aria_synthetic_envs_mcmc_3dgs_new"
local_root = Path("/root/autodl-tmp/gaussiangpt_data/raw/aria_subset")

scene_ids = [f"{i:05d}" for i in range(5, 31)]

files_per_scene = [
    "ckpts/point_cloud_30000.ply",
    "transforms_train.json",
    "stats/val_step30000.json",
]

for scene_id in scene_ids:
    print(f"\nDownloading scene {scene_id}", flush=True)

    for rel_file in files_per_scene:
        filename = f"{scene_id}/{rel_file}"
        print("  ", filename, flush=True)

        hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            filename=filename,
            local_dir=local_root,
            token=os.environ["HF_TOKEN"],
        )

print("\nDone.")
print("Saved to:", local_root)