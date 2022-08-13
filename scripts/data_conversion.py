import json
import os

import cv2 as cv
import numpy as np
from fire import Fire
from sn_calibration.baseline_cameras import \
    estimate_homography_from_line_correspondences
from sn_calibration.soccerpitch import SoccerPitch


def main(
    annotation_path,
    image_path,
    annotation_output=None,
    image_output=None,
):
    """Calculate homography (image to template) from soccernet annotation."""

    assert os.path.exists(annotation_path)
    assert os.path.exists(image_path)

    sp = SoccerPitch()
    img = cv.imread(image_path)

    with open(annotation_path, "r") as f:
        annotations = json.load(f)

    lines = []
    for line_class in sp.lines_classes:
        template_line = sp.get_2d_homogeneous_line(line_class)
        if line_class not in annotations.keys() or template_line is None:
            continue

        annot_points = annotations[line_class]
        p1 = np.array(
            [
                annot_points[0]["x"] * img.shape[1],
                annot_points[0]["y"] * img.shape[0],
                1,
            ],
            dtype="float",
        )
        p2 = np.array(
            [
                annot_points[1]["x"] * img.shape[1],
                annot_points[1]["y"] * img.shape[0],
                1,
            ],
            dtype="float",
        )
        annot_line = np.cross(p1, p2)

        lines.append((annot_line, template_line))

    ret, H = estimate_homography_from_line_correspondences(lines)
    print(ret, H, img.shape)

    if annotation_output is not None:
        np.save(annotation_output, H)
    if image_output is not None:
        warped_img = cv.warpPerspective(img, H, (img.shape[1], img.shape[0]))
        cv.imwrite(image_output, warped_img)


if __name__ == "__main__":
    Fire(main)
