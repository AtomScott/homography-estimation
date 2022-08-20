import argparse
import numpy as np
from tqdm.rich import tqdm
import cv2
from pathlib import Path
from dlhe.utils import get_keypoint_mesh, stacked_plot, project_keypoints


def prepare_worldcup(dataset_dir, output_dir, visualize=False, nx=13, ny=7):
    dataset_dir = Path(dataset_dir)
    output_dir = Path(output_dir)

    assert dataset_dir.exists(), f"{dataset_dir} does not exist"

    H_paths = sorted(dataset_dir.glob("**/*.homographyMatrix"))
    im_paths = sorted(dataset_dir.glob("**/*.jpg"))

    assert len(H_paths) == len(im_paths), f"{len(H_paths)} != {len(im_paths)}"

    length, width = 115, 74  # TODO: Contact author to see if this is correct
    keypoints = get_keypoint_mesh(nx=nx, ny=7, length=length, width=width)

    for H_path, im_path in zip(tqdm(H_paths), im_paths):
        H = np.loadtxt(H_path)
        im = cv2.imread(str(im_path))

        save_path = output_dir / im_path.parent.name / f"{im_path.stem}"
        save_path.parent.mkdir(exist_ok=True, parents=True)

        projected_keypoints = project_keypoints(keypoints, H)
        cv2.imwrite(str(save_path.with_suffix(".png")), im)
        np.save(str(save_path.with_suffix(".npy")), projected_keypoints)

        if visualize:
            im_stack = stacked_plot(im, H, keypoints, length, width)
            cv2.imwrite(str(save_path.with_suffix(".viz.png")), im_stack)

def prepare_ts_worldcup(dataset_dir, output_dir, visualize=False, nx=13, ny=7):
    raise NotImplementedError

def prepare_soccernet(dataset_dir, output_dir, visualize=False, nx=13, ny=7):
    raise NotImplementedError

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset_dir", help="Path to dataset", type=str, required=True
    )
    parser.add_argument(
        "--output_dir",
        help="Path to output directory",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--dataset",
        help="Type of dataset",
        type=str,
        required=True,
        choices=["worldcup", "ts_worldcup", "soccernet"],
    )
    parser.add_argument(
        "--visualize",
        "--plot",
        action="store_true",
        default=False,
    )

    args = parser.parse_args()

    if args.dataset.lower() == "worldcup":
        prepare_worldcup(args.dataset_dir, args.output_dir, args.visualize)
    elif args.dataset.lower() == "ts_worldcup":
        prepare_ts_worldcup(args.dataset_dir, args.output_dir, args.visualize)
    elif args.dataset.lower() == "soccernet":
        prepare_soccernet(args.dataset_dir, args.output_dir, args.visualize)
    else:
        raise NotImplementedError(f"dataset: {args.dataset} not implemented")
