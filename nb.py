# %%
import copy
import os
import glob

import cv2
import numpy as np
import pandas as pd
from cv2.typing import MatLike
from matplotlib import pyplot as plt

# %%
data_dir = "data"
dapi_list = sorted(glob.glob("*DAPI*.tif", root_dir=data_dir))
gfp_list = sorted(glob.glob("*GFP*.tif", root_dir=data_dir))
tritc_list = sorted(glob.glob("*TRITC*.tif", root_dir=data_dir))
print(f'Found {len(dapi_list)} DAPI, {len(gfp_list)} GFP, {len(tritc_list)} TRITC files')
print(dapi_list, gfp_list, tritc_list)

image_index = 1
dapi = cv2.imread(os.path.join(data_dir, dapi_list[image_index]))
dapi_gray = cv2.cvtColor(dapi, cv2.COLOR_BGR2GRAY)
print(dapi_gray.shape, dapi_gray.dtype)

gfp = cv2.imread(os.path.join(data_dir,gfp_list[image_index]))
gfp_gray = cv2.cvtColor(gfp, cv2.COLOR_BGR2GRAY)

tritc = cv2.imread(os.path.join(data_dir,tritc_list[image_index]))
tritc_gray = cv2.cvtColor(tritc, cv2.COLOR_BGR2GRAY)
# %%
plt.imshow(dapi_gray, cmap='gray')
plt.axis('off')
# np.unique(img)

# %%
_, mask = cv2.threshold(dapi_gray, 0, 255, cv2.THRESH_OTSU)

kernel = cv2.getStructuringElement(shape=cv2.MORPH_ELLIPSE, ksize=(7, 7))
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
mask = cv2.dilate(mask, kernel)

plt.imshow(mask, cmap='gray')
plt.axis('off')

# %%
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

print(f" Found {n} clusters")

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 20))
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

# %%
output_dir = 'out'
roi = dapi_list[image_index].split("_")[0]

num_bins = 256
bin_edges = np.arange(num_bins + 1)
channels = {"GFP": gfp_gray, "TRITC": tritc_gray}

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

    output_filename = os.path.join(output_dir, f"{roi}_{channel_name}.csv")
    try:
        final_df.to_csv(output_filename, index=False)
        print(
            f"  Successfully saved {channel_name} histogram data to {output_filename}"
        )
    except Exception as e:
        print(
            f"  Error saving CSV file for {channel_name} to {output_filename}: {e}"
        )
