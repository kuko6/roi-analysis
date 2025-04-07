import copy
import glob
import os

import cv2
import numpy as np
import pandas as pd
from cv2.typing import MatLike
from matplotlib import pyplot as plt

DATA_DIR = "data"
OUTPUT_DIR = "out"
os.makedirs(OUTPUT_DIR, exist_ok=True)

PLOT = True


def preprocess(gray: MatLike):
    """
    Preprocesses a grayscale image to create a binary mask for cell segmentation.
    """
    # apply Otsu's thresholding
    # might not be the best, as the intensities sometimes look overtuned
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)

    # apply morphological opening to remove small objects and noise
    # while preserving the shape of larger objects (cell clusters)
    kernel = cv2.getStructuringElement(shape=cv2.MORPH_ELLIPSE, ksize=(7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.dilate(mask, kernel)

    return mask


def find_contours(dapi: MatLike, gfp: MatLike, tritc: MatLike, mask: MatLike):
    """
    Identifies and visualizes cellular clusters by finding contours in the mask image.
    """

    # copy to avoid modifying original images
    segmented_img = copy.deepcopy(dapi)
    target_img = copy.deepcopy(gfp)
    target_img2 = copy.deepcopy(tritc)

    # find contours around the cell clusters
    contours, h = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    n = 0
    cluster_contours: list[MatLike] = []

    all_areas = [cv2.contourArea(c) for c in contours]
    # min_area_percentile = 98.0
    # contour_threshold = np.percentile(all_areas, min_area_percentile)
    contour_threshold = 2000

    # process each contour, filtering by area
    for i, contour in enumerate(contours):
        area = all_areas[i]
        if area > contour_threshold:
            cluster_contours.append(contour)

            # draw contour on all three images
            colour = (0, 255, 0)
            cv2.drawContours(segmented_img, contours, i, colour, 3)
            cv2.drawContours(target_img, contours, i, colour, 3)
            cv2.drawContours(target_img2, contours, i, colour, 3)

            n += 1

    print(f"  Found {n} clusters")

    if PLOT:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
        ax1.imshow(cv2.cvtColor(segmented_img, cv2.COLOR_BGR2RGB))
        ax1.set_title("DAPI")
        ax1.axis("off")
        ax2.imshow(cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB))
        ax2.set_title("GFP")
        ax2.axis("off")
        ax3.imshow(cv2.cvtColor(target_img2, cv2.COLOR_BGR2RGB))
        ax3.set_title("TRITC")
        ax3.axis("off")
        plt.tight_layout(rect=(0.0, 0.03, 1.0, 0.95))
        plt.show()

    return cluster_contours


def build_histogram(roi: str, channels: dict[str, MatLike], contours: list[MatLike]):
    """
    Calculates pixel intensity histograms for each cluster
    and saves the results to CSV files.
    """
    num_bins = 256
    bin_edges = np.arange(num_bins + 1)

    # nested dictionary that stores histogram data for each channel
    # {'ChannelName': {'Cluster_0': counts_array, 'Cluster_1': counts_array, ...}}
    channel_histograms = {name: {} for name in channels.keys()}

    # iterate over each cluster
    for i, cluster in enumerate(contours):
        # binary mask for the current cluster
        cluster_map = np.zeros_like(next(iter(channels.values())), dtype=np.uint8)
        cv2.drawContours(cluster_map, [cluster], 0, [255.0], thickness=cv2.FILLED)

        # boolean mask for pixel selection
        mask_boolean = cluster_map > 0

        # iterate over each grayscale image
        for channel_name, gray_image in channels.items():
            # select pixels from the current grayscale image that fall within the cluster mask
            pixels_inside_contour = gray_image[mask_boolean]

            # calculate histogram
            if pixels_inside_contour.size > 0:
                histogram_counts, _ = np.histogram(
                    pixels_inside_contour, bins=bin_edges
                )
                column_name = f"Cluster_{i}"
                channel_histograms[channel_name][column_name] = histogram_counts

    # save the histogram data
    print(f"\nSaving histogram data for ROI: {roi}")
    for channel_name, histograms_dict in channel_histograms.items():
        if not histograms_dict:
            print(f"  Skipping channel {channel_name}: No cluster data found.")
            continue

        final_df = pd.DataFrame(histograms_dict)
        final_df.insert(0, "Intensity", np.arange(num_bins))

        output_filename = os.path.join(OUTPUT_DIR, f"{roi}_{channel_name}.csv")
        try:
            final_df.to_csv(output_filename, index=False)
            print(
                f"  Successfully saved {channel_name} histogram data to {output_filename}"
            )
        except Exception as e:
            print(
                f"  Error saving CSV file for {channel_name} to {output_filename}: {e}"
            )


if __name__ == "__main__":
    dapi_list = sorted(glob.glob("*DAPI*.tif", root_dir=DATA_DIR))
    gfp_list = sorted(glob.glob("*GFP*.tif", root_dir=DATA_DIR))
    tritc_list = sorted(glob.glob("*TRITC*.tif", root_dir=DATA_DIR))
    print(f'Found {len(dapi_list)} DAPI, {len(gfp_list)} GFP, {len(tritc_list)} TRITC files')
    # print(dapi_list, gfp_list, tritc_list)

    for dapi_path, gfp_path, tritc_path in zip(dapi_list, gfp_list, tritc_list):
        roi = dapi_path.split("_")[0]

        dapi = cv2.imread(os.path.join(DATA_DIR, dapi_path))
        dapi_gray = cv2.cvtColor(dapi, cv2.COLOR_BGR2GRAY)
        # print(dapi_gray.shape, dapi_gray.dtype)

        gfp = cv2.imread(os.path.join(DATA_DIR, gfp_path))
        gfp_gray = cv2.cvtColor(gfp, cv2.COLOR_BGR2GRAY)

        tritc = cv2.imread(os.path.join(DATA_DIR, tritc_path))
        tritc_gray = cv2.cvtColor(tritc, cv2.COLOR_BGR2GRAY)

        print(f"\nAnalysing: {roi}")

        print("Step 1: Preprocessing DAPI image...")
        mask = preprocess(dapi_gray)

        print("Step 2: Finding contours...")
        clusters = find_contours(dapi, gfp, tritc, mask)

        if clusters:
            print("Step 3: Building GFP and TRITC histograms for clusters...")
            channels = {"GFP": gfp_gray, "TRITC": tritc_gray}
            build_histogram(roi, channels, clusters)
        else:
            print("Skipping histogram generation as no clusters met the criteria.")

        print('---------------------------------------')

    print("\nDone :D")
