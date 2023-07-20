import os
import numpy as np
import cv2
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap
import argparse


# ======================================== MEANSHIFT ALGORITHM ========================================


THRESHOLD = 1


def findpeak_opt(data: np.ndarray, idx, r, c):
    data_num = len(data)
    cpts = np.zeros((data_num,))
    center = data[idx]

    while True:
        # SPEEDUP 2
        # Collect points within distance r/c from the search path
        cpts[np.linalg.norm(data - center, axis=1) <= r / c] = 1

        # Collect points within radius
        points = data[np.linalg.norm(data - center, axis=1) <= r]

        # Compute mean of points within radius
        mean = np.mean(points, axis=0)

        # Check if shift > THRESHOLD
        if np.linalg.norm(center - mean) > THRESHOLD:
            center = mean
        else:
            break

    return center, cpts


def meanshift_opt(data: np.ndarray, r, c):
    data_num = len(data)
    labels = np.zeros((data_num,)) - 1
    peaks = []

    pbar = tqdm(total=data_num)
    labeled_num = 0
    while True:
        i = np.argmax(labels < 0)
        peak, cpts = findpeak_opt(data, i, r, c)

        labeled = False
        for prev_label, prev_peak in enumerate(peaks):
            if np.linalg.norm(peak - prev_peak) <= r / 2:
                label = prev_label
                peak = prev_peak
                labeled = True
                break

        if not labeled:
            label = len(peaks)
            peaks.append(peak)

        # Change label of i'th data point
        labels[i] = label

        # SPEEDUP 1
        # Change the label of points within a distance of r from the peak
        labels[np.linalg.norm(data - peak, axis=1) <= r] = label

        # SPEEDUP 2
        # Change the label of points within a distance of r/c from the search path
        labels[cpts > 0] = label

        # Update progress bar
        new_labeled_num = np.sum(labels >= 0)
        pbar.update(new_labeled_num - labeled_num)
        labeled_num = new_labeled_num

        # Check if all data has been labelled
        if labeled_num == data_num:
            break

    return labels, peaks


# ======================================== IMAGE PRE-PROCESSING ========================================


def load_image_lab(path):
    image_bgr = np.float32(cv2.imread(path)) / 255
    image_lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)

    return image_lab


def image_to_3d_vector(image: np.ndarray):
    h, w, c = image.shape
    vec_image = image.reshape((h * w, c))

    return vec_image


def image_to_5d_vector(image: np.ndarray):
    h, w, c = image.shape
    vec_image = image.reshape((h * w, c))
    coords = np.array([[(i, j) for j in range(w)] for i in range(h)]).reshape(
        (h * w, 2)
    )
    vec_image_xy = np.hstack((vec_image, coords))

    return vec_image_xy


# ======================================== IMAGE SEGMENTATION ========================================


def segmIm(path, vectorize_fun, r, c, verbose=False):
    """
    A function that uses the meanshift algorithm to perform image segmentation.
    """
    image = load_image_lab(path)
    vec_image = vectorize_fun(image)
    labels, peaks = meanshift_opt(vec_image, r, c)

    # Print results
    if verbose:
        print(f"##### Segmentation results for {path} #####")
        print()
        print(
            f"Parameters: {vectorize_fun.__name__[9:11].upper()} feature vector, r = {r}, c = {c}"
        )
        print()
        print("------------------------------------------------")
        print()
        print("Number of clusters found:", len(peaks))
        print()
        print("Labels -> Peaks:")
        for i, peak in enumerate(peaks):
            print(
                f"{str(i).rjust(6)} -> [ {'  '.join(f'{x:0.3f}'.rjust(8) for x in peak)} ]"
            )

    return labels, peaks


# ======================================== VISUALISING THE SEGMENTED IMAGE ========================================


def segm_image_peaks(
    im_shape: tuple, labels: np.ndarray, peaks: list[np.ndarray], save_as=None
):
    """
    A function to color each pixel according to the peak of its associated cluster.
    """
    # Reshape labels to the shape of the original image
    labels = labels.reshape(im_shape)

    # Convert the CIELAB peaks to RGB colors
    colors_lab = np.float32([[peak[:3]] for peak in peaks])
    colors = cv2.cvtColor(colors_lab, cv2.COLOR_LAB2RGB)

    # Create colormap
    cmap = ListedColormap(colors)

    # Create and plot segmented image
    labels_num = len(peaks)
    if labels_num > 1:
        labels /= np.max(labels)
    rgb_image = cmap(labels)
    plt.imshow(rgb_image)
    plt.axis("off")

    if save_as:
        plt.savefig(save_as, bbox_inches="tight")

    # plt.show()


def segm_image_clusters(
    im_shape: tuple, labels: np.ndarray, peaks: list[np.ndarray], save_as=None
):
    """
    A function to better visualise the distinct clusters (helpful in the case
    of 5D features).
    """
    # Reshape labels to the shape of the original image
    labels = labels.reshape(im_shape)

    # Create colormap
    labels_num = len(peaks)
    cmap = cm.get_cmap("Greys", labels_num)

    # Create and plot segmented image
    if labels_num > 1:
        labels /= np.max(labels)
    rgb_image = cmap(labels)
    plt.imshow(rgb_image)
    plt.axis("off")

    if save_as:
        plt.savefig(save_as, bbox_inches="tight")

    # plt.show()


# ======================================== MAIN ========================================


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-image", type=str, help="path to image")
    parser.add_argument(
        "-r", type=float, default=12, help="radius of meanshift algorithm"
    )
    parser.add_argument(
        "-c",
        type=float,
        default=4,
        help="associate points within a distance of r/c to the converged peak",
    )
    parser.add_argument(
        "-feature_type", type=str, default="3D", help="dimensions of the pixel features"
    )
    args = parser.parse_args()

    path, r, c = args.image, args.r, args.c

    if args.feature_type == "3D":
        vectorize_fun = image_to_3d_vector
    else:
        vectorize_fun = image_to_5d_vector

    return path, r, c, vectorize_fun


def main():
    path, r, c, vectorize_fun = parse_args()

    if path:
        h, w, _ = load_image_lab(path).shape
        feature_type = vectorize_fun.__name__[9:11].upper()

        print(f"{path}, {feature_type}, r={r}, c={c}")

        # Perform and time segmentation
        start = time.time()
        labels, peaks = segmIm(path, vectorize_fun, r, c)
        end = time.time()

        print("Clusters found:", len(peaks))
        print(f"Total time: {end - start:0.2f}s")

        # Plot and save peaks
        filename_peaks = f"{path}_{feature_type}_r={r}_c={c}.jpg"
        segm_image_peaks((h, w), labels, peaks, save_as=filename_peaks)

        # Plot and save clusters
        filename_clusters = f"{path}_{feature_type}_clusters_r={r}_c={c}.jpg"
        segm_image_clusters((h, w), labels, peaks, save_as=filename_clusters)

    else:
        print(
            f"Usage: python {os.path.basename(__file__)} -image [-r] [-c] [-feature_type]"
        )


if __name__ == "__main__":
    main()
