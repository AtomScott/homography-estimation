import cv2 as cv
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from fire import Fire


def get_keypoint_mesh(length=105, width=76):

    nx, ny = (13, 8)
    x = np.linspace(0, length, nx)
    y = np.linspace(0, width, ny)

    xv, yv = np.meshgrid(x, y, indexing="ij")
    xv = xv.flatten()
    yv = yv.flatten()

    return xv, yv


def main(H_path, image_path):
    H = np.load(H_path)
    img = cv.imread(image_path)

    xv, yv = get_keypoint_mesh()
    points = np.array([np.stack((xv, yv), axis=1)])
    xv, yv = cv.perspectiveTransform(points, np.linalg.inv(H))[0].T

    viridis = matplotlib.cm.get_cmap("jet")
    n_dot = 0
    norm = matplotlib.colors.Normalize(vmin=0, vmax=points.shape[1])
    for xi, yi in zip(xv, yv):
        color = [int(c * 255) for c in viridis(norm(n_dot))[:3]]
        cv.circle(img, (int(xi), int(yi)), 5, color, -1)
        n_dot += 1
    cv.imwrite("keypoints.png", img)


if __name__ == "__main__":
    Fire(main)
