"""DataLoader used to train the segmentation network used for the prediction of extremities.
"""

from argparse import ArgumentParser
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from pytorch_lightning.utilities.types import _PATH, Any, Dict, Union
from rich import inspect
from torch import tensor
from torch.utils.data import Dataset
from tqdm.rich import tqdm


class KeypointDataset(Dataset):
    def __init__(
        self, data_dir: _PATH, sigma: int = 10, scale: int = 1, transform=None
    ):
        """Dataset for keypoints.

        Args:
            data_dir (_PATH): Path to the directory containing the images and keypoints.
            sigma (int, optional): _description_. Defaults to 10.
            scale (int, optional): _description_. Defaults to 1.
        """
        self._gaussians: Dict[Any, Any] = {}

        self.data_dir = Path(data_dir)
        self.scale = scale
        self.sigma = sigma
        self.transform = transform

        self.img_paths = sorted(self.data_dir.glob("**/*[!.viz].png"))
        self.kp_paths = sorted(self.data_dir.glob("**/*.npy"))
        assert len(self.img_paths) == len(
            self.kp_paths
        ), f"{len(self.img_paths)} != {len(self.kp_paths)}"

    def __len__(self) -> int:
        return len(self.img_paths)

    def preprocess_image(self, img) -> np.ndarray:
        w, h = img.size
        _h = int(h * self.scale)
        _w = int(w * self.scale)
        assert _w > 0
        assert _h > 0

        _img = img.resize((_w, _h))
        _img = np.array(_img)
        if len(_img.shape) == 2:  ## gray/mask images
            _img = np.expand_dims(_img, axis=-1)

        # hwc to chw
        _img = _img.transpose((2, 0, 1))
        if _img.max() > 1:
            _img = _img / 255.0
        return _img

    def preprocess_keypoints(self, keypoints):
        scale = self.scale
        return keypoints * scale

    def generate_gaussian(self, h, w, mu_x, mu_y):
        """
        Generates a 2D Gaussian point at location x,y in tensor t.

        x should be in range (-1, 1) to match the output of fastai's PointScaler.

        sigma is the standard deviation of the generated 2D Gaussian.
        """
        sigma = self.sigma * self.scale
        _gaussians = self._gaussians
        t = np.zeros((h, w))

        tmp_size = sigma * 3

        # Top-left
        x1, y1 = int(mu_x - tmp_size), int(mu_y - tmp_size)

        # Bottom right
        x2, y2 = int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)
        if x1 >= w or y1 >= h or x2 < 0 or y2 < 0:
            return t

        size = 2 * tmp_size + 1
        tx = np.arange(0, size, 1, np.float32)
        ty = tx[:, np.newaxis]
        x0 = y0 = size // 2

        # The gaussian is not normalized, we want the center value to equal 1
        g = (
            _gaussians[sigma]
            if sigma in _gaussians
            else np.exp(-((tx - x0) ** 2 + (ty - y0) ** 2) / (2 * sigma ** 2))
        )
        _gaussians[sigma] = g

        # Determine the bounds of the source gaussian
        g_x_min, g_x_max = max(0, -x1), min(x2, w) - x1
        g_y_min, g_y_max = max(0, -y1), min(y2, h) - y1

        # Image range
        img_x_min, img_x_max = max(0, x1), min(x2, w)
        img_y_min, img_y_max = max(0, y1), min(y2, h)

        t[img_y_min:img_y_max, img_x_min:img_x_max] = g[
            g_y_min:g_y_max, g_x_min:g_x_max
        ]
        return t

    def __getitem__(self, i):
        # image = self.preprocess_image(Image.open(self.img_paths[i]))

        image = cv2.imread(str(self.img_paths[i]))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        kps = self.preprocess_keypoints(np.load(self.kp_paths[i]).T)

        mask = []
        for kp in kps:
            h, w = image.shape[:2]
            x, y = kp
            heatmap = self.generate_gaussian(h, w, x, y)
            mask.append(heatmap)
        mask = np.array(mask).transpose((1, 2, 0))

        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]

        image = torch.from_numpy(image.transpose(2, 0, 1)).float()
        mask = torch.from_numpy(mask.transpose(2, 0, 1)).float()
        return image, mask


if __name__ == "__main__":

    # Load the arguments
    parser = ArgumentParser(description="dataloader")

    parser.add_argument(
        "--data_dir",
        default="./data/",
        type=str,
        help="Path to the data folder",
    )

    args = parser.parse_args()
    inspect(args)

    kp_dataset = KeypointDataset(args.data_dir)
    print(f"KeypointDataset: {len(kp_dataset)}")
    for im, masks in tqdm(kp_dataset):
        pass
