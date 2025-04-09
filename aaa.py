import copy
import os
import re
import argparse

import cv2
import numpy as np
import pandas as pd
from cv2.typing import MatLike
from matplotlib import pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--dapi", type=str, required=True, help="path to dapi tif file")
parser.add_argument("--data", type=str, default="data", help="path to data dir containing images")
parser.add_argument("--plot", action='store_true', help="whether to plot intermediate and final steps") # Use action='store_true'
parser.add_argument("--out", type=str, default="out", help="path to output dir for csv files")
# Add a parameter for minimum cell size if needed
parser.add_argument("--min_area", type=int, default=50, help="minimum pixel area for a segmented cell")


args = parser.parse_args()

DATA_DIR = args.data
OUTPUT_DIR = args.out
os.makedirs(OUTPUT_DIR, exist_ok=True)

PLOT = args.plot
MIN_CELL_AREA = args.min_area

# --- Watershed Segmentation Function ---
def watershed_segmentation(dapi_gray: MatLike, dapi_color: MatLike):
    """
    Performs watershed segmentation on the DAPI image to separate touching nuclei.

    Args:
        dapi_gray: Grayscale DAPI image.
        dapi_color: Original color DAPI image (needed for cv2.watershed).

    Returns:
        markers: A labeled image where each unique positive integer represents a
                 segmented nucleus. Value -1 indicates watershed boundaries. Value 1 is background.
        vis_img: A visualization image with boundaries drawn.
    """
    print("  Performing Watershed Segmentation...")

    # 1. Thresholding (Otsu's) - Foreground is white
    _, thresh = cv2.threshold(dapi_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if PLOT:
        plt.figure(figsize=(8, 4))
        plt.subplot(121); plt.imshow(dapi_gray, cmap='gray'); plt.title('Original DAPI'); plt.axis('off')
        plt.subplot(122); plt.imshow(thresh, cmap='gray'); plt.title('Otsu Threshold'); plt.axis('off')
        plt.suptitle('Step 1: Thresholding')
        plt.tight_layout(rect=(0, 0, 1, 0.95))
        plt.show()

    # 2. Noise removal (Optional but often helpful)
    # Use opening to remove small noise spots
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    # 3. Sure background area
    # Dilating sure removes small holes, sure background is black area expanded
    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    # 4. Sure foreground area
    # Distance transform finds distance to closest zero pixel (background)
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    # Threshold the distance map to get peaks == sure foreground
    # Adjust the multiplier (e.g., 0.5 - 0.7) based on results
    ret, sure_fg = cv2.threshold(dist_transform, 0.6 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg) # Convert to uint8

    if PLOT:
        plt.figure(figsize=(12, 4))
        plt.subplot(131); plt.imshow(opening, cmap='gray'); plt.title('Opening'); plt.axis('off')
        plt.subplot(132); plt.imshow(dist_transform, cmap='magma'); plt.title('Distance Transform'); plt.axis('off')
        plt.subplot(133); plt.imshow(sure_fg, cmap='gray'); plt.title('Sure Foreground'); plt.axis('off')
        plt.suptitle('Steps 2-4: Morphological Ops & Distance Transform')
        plt.tight_layout(rect=(0, 0, 1, 0.95))
        plt.show()

    # 5. Unknown region
    unknown = cv2.subtract(sure_bg, sure_fg)

    # 6. Marker labelling
    ret, markers = cv2.connectedComponents(sure_fg)
    # Add one to all labels so that sure background is 1, not 0
    markers = markers + 1
    # Mark the region of unknown with zero
    markers[unknown == 255] = 0

    if PLOT:
        plt.figure(figsize=(8, 4))
        plt.subplot(121); plt.imshow(sure_bg, cmap='gray'); plt.title('Sure Background'); plt.axis('off')
        plt.subplot(122); plt.imshow(markers, cmap='jet'); plt.title('Markers (0=Unknown)'); plt.axis('off')
        plt.suptitle('Steps 5-6: Unknown Region & Markers')
        plt.tight_layout(rect=(0, 0, 1, 0.95))
        plt.show()

    # 7. Apply watershed
    # Watershed needs the original image (color) and the markers
    markers = cv2.watershed(dapi_color, markers)
    # Watershed boundaries are marked with -1

    # Create a visualization image
    vis_img = dapi_color.copy()
    vis_img[markers == -1] = [255, 0, 0]  # Draw boundaries in red

    # Count segments (excluding background label 1 and boundaries -1)
    unique_labels = np.unique(markers)
    num_segments = len(unique_labels[unique_labels > 1])
    print(f"  Watershed found {num_segments} potential segments (before area filtering).")

    return markers, vis_img


# --- Modify build_histogram to use markers ---
def build_histogram(channels: dict[str, MatLike], markers: MatLike, roi_name: str):
    """
    Calculates pixel intensity histograms for each segmented nucleus (marker label)
    and saves the results to CSV files. Filters segments by minimum area.

    Args:
        channels: Dictionary {'ChannelName': grayscale_image}.
        markers: Labeled image from watershed segmentation.
        roi_name: Name of the region of interest for file naming.
    """
    num_bins = 256
    bin_edges = np.arange(num_bins + 1)

    # Nested dictionary that stores histogram data for each channel
    # {'ChannelName': {'Segment_ID': counts_array, ...}}
    channel_histograms = {name: {} for name in channels.keys()}

    # Get unique marker labels (potential nuclei segments)
    # Ignore background (label 1) and watershed boundaries (label -1)
    unique_labels = np.unique(markers)
    segment_labels = unique_labels[unique_labels > 1]

    print(f"\nProcessing {len(segment_labels)} potential segments for histograms...")
    valid_segment_count = 0

    # Iterate over each potential segment label
    for label in segment_labels:
        # Create a boolean mask for the current segment
        mask_boolean = (markers == label)

        # --- Area Filtering ---
        area = np.sum(mask_boolean)
        if area < MIN_CELL_AREA:
            # print(f"  Skipping segment {label}: Area {area} < {MIN_CELL_AREA}")
            continue # Skip small segments

        valid_segment_count += 1
        segment_name = f"Segment_{label}" # Use label as unique ID

        # Iterate over each grayscale image channel (GFP, TRITC)
        for channel_name, gray_image in channels.items():
            # Select pixels from the current grayscale image that fall within the segment mask
            pixels_inside_segment = gray_image[mask_boolean]

            # Calculate histogram
            if pixels_inside_segment.size > 0:
                histogram_counts, _ = np.histogram(
                    pixels_inside_segment, bins=bin_edges
                )
                channel_histograms[channel_name][segment_name] = histogram_counts
            else:
                 # Handle case where segment mask might be empty after some operation (shouldn't happen here)
                 channel_histograms[channel_name][segment_name] = np.zeros(num_bins, dtype=int)

    print(f"  Kept {valid_segment_count} segments after area filtering (Min Area = {MIN_CELL_AREA}).")

    if valid_segment_count == 0:
        print(f"  No segments met the minimum area criteria for ROI: {roi_name}. Skipping histogram saving.")
        return

    # Save the histogram data
    print(f"\nSaving histogram data for ROI: {roi_name}")
    for channel_name, histograms_dict in channel_histograms.items():
        if not histograms_dict:
            # This check might be redundant now due to valid_segment_count check above
            print(f"  Skipping channel {channel_name}: No valid segment data found.")
            continue

        final_df = pd.DataFrame(histograms_dict)
        final_df.insert(0, "Intensity", np.arange(num_bins))

        output_filename = os.path.join(OUTPUT_DIR, f"{roi_name}_{channel_name}.csv")
        try:
            final_df.to_csv(output_filename, index=False)
            print(
                f"  Successfully saved {channel_name} histogram data to {output_filename}"
            )
        except Exception as e:
            print(
                f"  Error saving CSV file for {channel_name} to {output_filename}: {e}"
            )


# --- Main Script Logic ---
if __name__ == "__main__":
    dapi_path_short = args.dapi # Just the filename

    # Regex to extract parts and construct other filenames
    pattern = re.compile(r"^(.*_)(\d)(_.*_Confocal )([A-Z]+)(_.*\.tif)$")
    match = pattern.match(dapi_path_short)

    if match:
        prefix = match.group(1)
        middle = match.group(3)
        suffix = match.group(5)
        # Assuming DAPI channel number is 2
        gfp_path_short = f"{prefix}3{middle}GFP{suffix}"
        tritc_path_short = f"{prefix}4{middle}TRITC{suffix}"
        roi_name = dapi_path_short.split("_")[0] # Extract ROI name
        print(f"Derived ROI Name: {roi_name}")
        print(f"DAPI file: {dapi_path_short}")
        print(f"GFP file: {gfp_path_short}")
        print(f"TRITC file: {tritc_path_short}")

    else:
        print(f"Error: The input DAPI filename '{dapi_path_short}' does not match the expected pattern.")
        print("Expected pattern example: 'ROI_Name_2_..._Confocal DAPI_....tif'")
        quit()

    # Construct full paths
    dapi_full_path = os.path.join(DATA_DIR, dapi_path_short)
    gfp_full_path = os.path.join(DATA_DIR, gfp_path_short)
    tritc_full_path = os.path.join(DATA_DIR, tritc_path_short)

    # Load images
    dapi = cv2.imread(dapi_full_path)
    gfp = cv2.imread(gfp_full_path)
    tritc = cv2.imread(tritc_full_path)

    if dapi is None:
        print(f"Error loading DAPI image: {dapi_full_path}")
        quit()
    if gfp is None:
        print(f"Error loading GFP image: {gfp_full_path}")
        quit()
    if tritc is None:
        print(f"Error loading TRITC image: {tritc_full_path}")
        quit()

    # Convert to grayscale for processing
    dapi_gray = cv2.cvtColor(dapi, cv2.COLOR_BGR2GRAY)
    gfp_gray = cv2.cvtColor(gfp, cv2.COLOR_BGR2GRAY)
    tritc_gray = cv2.cvtColor(tritc, cv2.COLOR_BGR2GRAY)

    print(f"\nAnalysing ROI: {roi_name}")

    print("Step 1: Performing Watershed Segmentation on DAPI...")
    # Pass both gray and color DAPI to watershed
    markers, watershed_vis = watershed_segmentation(dapi_gray, dapi)

    # Optional: Plot final segmentation result
    if PLOT:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)
        fig.canvas.manager.set_window_title(f"{roi_name} - Watershed Results")
        axes[0].imshow(cv2.cvtColor(dapi, cv2.COLOR_BGR2RGB))
        axes[0].set_title("Original DAPI")
        axes[0].axis("off")
        # Use a color map (like jet or tab20) for markers, setting background (1) and boundaries (-1) explicitly
        cmap = plt.cm.jet
        cmap.set_under(color='black') # For -1 boundaries (or just use the vis_img)
        cmap.set_bad(color='white')   # For background (label 1) - might need adjustment
        # Or simply show the vis_img
        axes[1].imshow(cv2.cvtColor(watershed_vis, cv2.COLOR_BGR2RGB))
        axes[1].set_title("Watershed Boundaries (Red)")
        axes[1].axis("off")
        axes[2].imshow(markers, cmap='tab20b') # 'tab20b' provides distinct colors for many segments
        axes[2].set_title("Labeled Segments")
        axes[2].axis("off")
        plt.tight_layout()
        plt.show()

    print("Step 2: Building GFP and TRITC histograms for segmented nuclei...")
    channels = {"GFP": gfp_gray, "TRITC": tritc_gray}
    # Pass the markers image instead of contours
    build_histogram(channels, markers, roi_name)

    print('---------------------------------------')
    print("\nDone :D")
