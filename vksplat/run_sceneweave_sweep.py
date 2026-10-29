from pathlib import Path
import json
from simple_trainer import TrainerConfig, train_main

DATASET = Path(
    "/mnt/cps_persistent1_shared/datasets/public/"
    "mipnerf360_v2_dataset/room"
)

OUTPUT = Path.home() / "git/sceneweave/outputs/vksplat"

for steps in [3000, 10000, 30000]:
    config = TrainerConfig(
        dataset_dir=str(DATASET),
        image_dir="images_2",
        sparse_dir="sparse/0",
        output_dir=str(OUTPUT / f"steps_{steps}"),
        train_steps=steps,
        max_steps=30000,
        enable_viewer=False,
        save_train_renders=False,
    )

    print(f"\n{'=' * 60}")
    print(f"Training {steps} iterations")
    print(f"{'=' * 60}\n", flush=True)

    train_main(config)
