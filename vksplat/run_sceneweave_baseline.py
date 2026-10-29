from pathlib import Path
from simple_trainer import TrainerConfig, train_main

DATASET = Path(
    "/mnt/cps_persistent1_shared/datasets/public/"
    "mipnerf360_v2_dataset/room"
)

assert (DATASET / "sparse/0/cameras.bin").exists()
assert (DATASET / "sparse/0/images.bin").exists()
assert (DATASET / "images_2").is_dir()

config = TrainerConfig(
    dataset_dir=str(DATASET),
    image_dir="images_2",
    sparse_dir="sparse/0",
    output_dir=str(
        Path.home() / "git/sceneweave/outputs/vksplat"
    ),
    train_steps=1000,
    max_steps=30000,
    enable_viewer=False,
    save_train_renders=False,
)

print("Dataset:", config.dataset_dir)
print("Output:", config.output_dir)
print("Training steps:", config.train_steps)

train_main(config)
