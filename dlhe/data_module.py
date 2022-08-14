from argparse import ArgumentParser
from pathlib import Path
from typing import *

import albumentations as A
import pytorch_lightning as pl
from rich import inspect
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm.rich import tqdm

from dlhe.dataset import KeypointDataset


class KeypointDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str = "path/to/dir",
        batch_size: int = 8,
        pin_memory: bool = False,
        num_workers: int = 1,
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.check_data_dir()

        self.batch_size = batch_size
        self.pin_memory = pin_memory
        self.num_workers = num_workers
        self.transform = A.Compose(
            [
                A.Affine(p=0.5),
                A.RandomResizedCrop(width=256, height=256, scale=(0.8, 1.0)),
                A.RandomShadow(p=0.2),
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, p=0.3),
            ]
        )

    def check_data_dir(self):
        data_dir = self.data_dir
        assert Path(data_dir).exists(), f"{data_dir} does not exist"
        found_datasets = {
            dataset: (data_dir / dataset).exists()
            for dataset in ["train", "val", "test"]
        }
        if not all(found_datasets.values()):
            for dataset, found in found_datasets.items():
                if not found:
                    print(f"{dataset} not found in {data_dir}")
            print(
                f"Data directory {data_dir} must contain the following datasets: {found_datasets.keys()} to be recognized by this module"
            )

    def setup(self, stage: Optional[str] = None):
        data_dir = self.data_dir
        self.trainset_path = Path(data_dir) / "train"
        self.valset_path = Path(data_dir) / "val"
        self.testset_path = Path(data_dir) / "test"

        self.trainset = KeypointDataset(self.trainset_path, transform=self.transform)
        self.valset = KeypointDataset(self.valset_path)
        self.testset = KeypointDataset(self.testset_path)

    def train_dataloader(self):
        return DataLoader(
            self.trainset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self):
        return DataLoader(
            self.valset,
            batch_size=1,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self):
        return DataLoader(
            self.testset,
            batch_size=1,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )


if __name__ == "__main__":

    # Load the arguments
    parser = ArgumentParser(description="dataloader")

    parser.add_argument(
        "--data_dir",
        default="./data/",
        type=str,
        help="Path to the data folder",
    )

    parser.add_argument(
        "--num_workers",
        default=1,
        type=int,
        help="Number of workers to use for the dataloaders",
    )
    args = parser.parse_args()
    inspect(args)

    dm = KeypointDataModule(args.data_dir, num_workers=args.num_workers)
    dm.setup()

    for im, masks in tqdm(dm.train_dataloader(), desc="train_dataloader"):
        pass

    for im, masks in tqdm(dm.val_dataloader(), desc="val_dataloader"):
        pass

    for im, masks in tqdm(dm.test_dataloader(), desc="test_dataloader"):
        pass
