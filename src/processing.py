import os
import re

import cv2
import numpy as np
import pandas as pd
from cv2.typing import MatLike
from matplotlib import pyplot as plt


class RoiAnalyser:
    """
    This class provides functionality for cell cluster segmentation, contour detection,
    and histogram analysis of microscopy images, specifically DAPI, GFP,
    and TRITC channels.
    """

    def __init__(
        self,
        data_dir="data",
        out_dir="out",
        plot=True,
        size_threshold=3000,
        opening_kernel=13,
        opening_iter=1,
        dilatation_kernel=7,
        dilatation_iter=2,
    ):
        """
        Initialize the RoiAnalyser with processing parameters.

        Args:
            data_dir (str, optional): Directory containing input images. Defaults to "data".
            out_dir (str, optional): Directory where results will be saved. Defaults to "out".
            plot (bool, optional): Whether to generate visualization plots. Defaults to True.
            size_threshold (int, optional): Minimum size for clusters in pixels. Defaults to 3000.
            opening_kernel (int, optional): Size of kernel for opening operations. Defaults to 13.
            opening_iter (int, optional): Number of iterations for opening. Defaults to 1.
            dilatation_kernel (int, optional): Size of kernel for dilation. Defaults to 7.
            dilatation_iter (int, optional): Number of iterations for dilation. Defaults to 2.
        """
        self.DATA_DIR = data_dir
        self.OUTPUT_DIR = out_dir
        self.PLOT = plot

        self.OPENING_KERNEL = opening_kernel
        self.OPENING_ITER = opening_iter
        self.DILATATION_ITER = dilatation_iter
        self.DILATATION_KERNEL = dilatation_kernel
        self.SIZE_THRESHOLD = size_threshold

        self.Z = ""
        self.ROI_NAME = ""

        os.makedirs(out_dir, exist_ok=True)

    def create_binary_mask(self, gray: MatLike):
        """
        Preprocesses a grayscale image to create a binary mask for cell segmentation.
        """
        # apply Otsu's thresholding
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)

        # apply morphological opening to remove small objects and noise
        # while preserving the shape of larger objects (cell clusters)
        kernel = cv2.getStructuringElement(
            shape=cv2.MORPH_ELLIPSE, ksize=(self.OPENING_KERNEL, self.OPENING_KERNEL)
        )
        mask = cv2.morphologyEx(
            mask, cv2.MORPH_OPEN, kernel, iterations=self.OPENING_ITER
        )

        # perform additional dilatation
        kernel = cv2.getStructuringElement(
            shape=cv2.MORPH_ELLIPSE,
            ksize=(self.DILATATION_KERNEL, self.DILATATION_KERNEL),
        )
        mask = cv2.dilate(mask, kernel, iterations=self.DILATATION_ITER)

        # plt.imshow(mask, cmap='gray')
        # plt.axis('off')

        return mask

    def define_clusters(self, mask: MatLike):
        """
        Identifies and visualizes clusters by finding contours in the mask image.
        """

        # find contours around the cell clusters
        contours, h = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # process each contour, filtering by area
        cluster_contours: list[MatLike] = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > self.SIZE_THRESHOLD:
                cluster_contours.append(contour)

        print(f"  Found {len(cluster_contours)} clusters")

        return cluster_contours

    def plot_contours(
        self, dapi: MatLike, gfp: MatLike, tritc: MatLike, contours: list[MatLike]
    ):
        """
        Plots found contours on all 3 images.
        """

        # draw contour on all three images
        for i, contour in enumerate(contours):
            colour = (0, 255, 0)
            cv2.drawContours(dapi, contours, i, colour, 3)
            cv2.drawContours(gfp, contours, i, colour, 3)
            cv2.drawContours(tritc, contours, i, colour, 3)

            colour = (0, 255, 255)
            font = cv2.FONT_HERSHEY_SIMPLEX
            text = str(i + 1)
            coords = tuple(contour[0][0])
            cv2.putText(dapi, text, coords, font, 5, colour, 10, cv2.LINE_AA)
            cv2.putText(gfp, text, coords, font, 5, colour, 10, cv2.LINE_AA)
            cv2.putText(tritc, text, coords, font, 5, colour, 10, cv2.LINE_AA)

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 10))
        ax1.imshow(cv2.cvtColor(dapi, cv2.COLOR_BGR2RGB))
        ax1.set_title("DAPI")
        ax1.axis("off")

        ax2.imshow(cv2.cvtColor(gfp, cv2.COLOR_BGR2RGB))
        ax2.set_title("GFP")
        ax2.axis("off")

        ax3.imshow(cv2.cvtColor(tritc, cv2.COLOR_BGR2RGB))
        ax3.set_title("TRITC")
        ax3.axis("off")

        plt.tight_layout(rect=(0.0, 0.03, 1.0, 0.95))
        plt.show()

        return fig

    def build_histogram(
        self,
        channels: dict[str, MatLike],
        contours: list[MatLike],
        z: str,
        channel_histograms: dict,
    ):
        """
        Calculates pixel intensity histograms for each cluster.
        """

        num_bins = 256
        bin_edges = np.arange(num_bins + 1)

        # nested dictionary that stores histogram data for each channel
        # {'ChannelName': {'Z4' : {'Cluster_0': counts_array, 'Cluster_1': counts_array, ...}, {'Z6': {'Cluster_0': counts_array, 'Cluster_1': counts_array, ...}}}}
        for name in channel_histograms.keys():
            channel_histograms[name][z] = {}
        # print(channel_histograms)

        # iterate over each cluster
        for i, cluster in enumerate(contours):
            # binary mask for the current cluster
            cluster_map = np.zeros_like(channels["GFP"], dtype=np.uint8)
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
                    column_name = f"Cluster_{i + 1}"
                    channel_histograms[channel_name][z][column_name] = histogram_counts

        return channel_histograms

    def save_histogram(self, channel_histograms, figure):
        """
        Saves computed histograms into csv files. Returns (gfp_df, tritc_df)
        """

        print(f"\nSaving histogram data for ROI: {self.ROI_NAME}")

        if figure:
            figure.savefig(
                os.path.join(self.OUTPUT_DIR, f"{self.ROI_NAME}_{self.Z}.png")
            )

        out_files = {}
        for channel_name, ch_data in channel_histograms.items():
            data_dict = {}
            for z_level, z_data in ch_data.items():
                data_dict[z_level] = pd.DataFrame(z_data)

            output_filename = os.path.join(
                self.OUTPUT_DIR, f"{self.ROI_NAME}_{channel_name}.csv"
            )
            try:
                final_df = pd.concat(data_dict, axis=1)
                pd.DataFrame.insert(
                    final_df, loc=0, column=("", "Pixel_values"), value=np.arange(256)
                )

                out_files[channel_name] = output_filename
                # out_files.append(output_filename)
                final_df.to_csv(output_filename, index=False)
                print(
                    f"  Successfully saved {channel_name} histogram data to {output_filename}"
                )
            except Exception as e:
                print(
                    f"  Error saving CSV file for {channel_name} to {output_filename}: {e}"
                )

        return out_files

    def parse_dapi_path(self, dapi_path: str):
        """
        Parse DAPI image path and return coresponding gfp and tritc paths.
        """

        pattern = re.compile(r"^(.*_)(\d)(_\d(.*)_Confocal )([A-Z]+)(_.*\.tif)$")
        match = pattern.match(dapi_path)
        if match is None:
            raise ValueError(
                f"Error: The input path '{dapi_path}' does not match the expected pattern."
            )

        prefix = match.group(1)
        middle = match.group(3)
        z = match.group(4)
        suffix = match.group(6)
        # print(prefix, middle, z, suffix)

        gfp_path = f"{prefix}3{middle}GFP{suffix}"
        tritc_path = f"{prefix}4{middle}TRITC{suffix}"

        print(dapi_path, gfp_path, tritc_path)
        roi_name = prefix.split("_")[0]

        self.ROI_NAME = roi_name
        self.Z = z

        return gfp_path, tritc_path

    def load_data(self, dapi_path: str, gfp_path: str, tritc_path: str):
        """
        Load data from the given paths and return a dictionary of images and their grayscale versions.
        """

        dapi = cv2.imread(os.path.join(self.DATA_DIR, dapi_path))
        dapi_gray = cv2.cvtColor(dapi, cv2.COLOR_BGR2GRAY)
        # print(dapi_gray.shape, dapi_gray.dtype)

        gfp = cv2.imread(os.path.join(self.DATA_DIR, gfp_path))
        gfp_gray = cv2.cvtColor(gfp, cv2.COLOR_BGR2GRAY)

        tritc = cv2.imread(os.path.join(self.DATA_DIR, tritc_path))
        tritc_gray = cv2.cvtColor(tritc, cv2.COLOR_BGR2GRAY)

        imgs = {
            "dapi": {"img": dapi, "gray": dapi_gray},
            "gfp": {"img": gfp, "gray": gfp_gray},
            "tritc": {"img": tritc, "gray": tritc_gray},
        }

        return imgs

    def repeat_for_additional_images(
        self, z: str, dapi_path: str, clusters: list[MatLike], channel_histograms
    ):
        """
        Repeat the analysis for additional images.
        """

        print("Repeating for: ", z)

        pattern = re.compile(r"^(.*_)(\d)(_\d(.*)_Confocal )([A-Z]+)(_.*\.tif)$")
        match = pattern.match(dapi_path)
        if match is None:
            raise ValueError(
                f"Error: The input path '{dapi_path}' does not match the expected pattern."
            )

        prefix = match.group(1)
        suffix = match.group(6)

        dapi_path = f"{prefix}2_1{z}_Confocal DAPI{suffix}"
        gfp_path = f"{prefix}3_1{z}_Confocal GFP{suffix}"
        tritc_path = f"{prefix}4_1{z}_Confocal TRITC{suffix}"
        print(dapi_path, gfp_path, tritc_path)

        dapi = cv2.imread(os.path.join(self.DATA_DIR, dapi_path))

        gfp = cv2.imread(os.path.join(self.DATA_DIR, gfp_path))
        gfp_gray = cv2.cvtColor(gfp, cv2.COLOR_BGR2GRAY)

        tritc = cv2.imread(os.path.join(self.DATA_DIR, tritc_path))
        tritc_gray = cv2.cvtColor(tritc, cv2.COLOR_BGR2GRAY)

        if self.PLOT:
            self.plot_contours(dapi, gfp, tritc, clusters)

        channels = {"GFP": gfp_gray, "TRITC": tritc_gray}
        channel_histograms = self.build_histogram(
            channels, clusters, z, channel_histograms
        )

        return channel_histograms

    def run_analysis(self, dapi_path: str):
        """
        Runs analysis for the first image group (DAPI, GFP and TRITC).
        """

        gfp_path, tritc_path = self.parse_dapi_path(dapi_path)
        imgs = self.load_data(dapi_path, gfp_path, tritc_path)

        print(f"\nAnalysing: {self.ROI_NAME}, {self.Z}")

        print("Step 1: Preprocessing DAPI image...")
        mask = self.create_binary_mask(imgs["dapi"]["gray"])

        print("Step 2: Finding contours...")
        clusters = self.define_clusters(mask)

        figure = None
        if self.PLOT:
            figure = self.plot_contours(
                imgs["dapi"]["img"], imgs["gfp"]["img"], imgs["tritc"]["img"], clusters
            )

        print("Step 3: Building GFP and TRITC histograms for clusters...")
        channels = {"GFP": imgs["gfp"]["gray"], "TRITC": imgs["tritc"]["gray"]}
        channel_histograms = {"GFP": {}, "TRITC": {}}

        channel_histograms = self.build_histogram(
            channels, clusters, self.Z, channel_histograms
        )

        return clusters, channel_histograms, figure

    def apopnec_ratio(self, file):
        """
        Runs some analysis idk.
        """

        df = pd.read_csv(file)
        df = df.drop(index=0).reset_index(drop=True)

        pattern = re.compile(r'^Z[2-6](?:\..+)?(_mult)?$')
        columns_to_keep = ['Unnamed: 0'] + [col for col in df.columns if pattern.match(col)]

        # Keep only those columns
        df = df[columns_to_keep]

        for col in df.columns[1:]:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df['Unnamed: 0'] = pd.to_numeric(df['Unnamed: 0'], errors='coerce')

        # df_multiplied = df
        z_columns = df.columns[1:]
        column_sums = df[z_columns].sum()

        df_multiplied = df.copy()
        for col in z_columns:
            df_multiplied[col + '_mult'] = df['Unnamed: 0'] * df[col]

        # Define the row index to start summing from (72 onwards)
        start_row = 72

        # Select Z columns to sum
        z_columns_mult = [col + '_mult' for col in z_columns]

        # Sum each column from row 72 onward
        column_sums_from_72 = df_multiplied.loc[start_row:, z_columns_mult].sum()

        # Display the result
        print(column_sums_from_72)

        # Sum original columns
        column_sums = df[z_columns].sum()

        # Sum multiplied columns from row 72 downward
        column_sums_from_72 = df_multiplied.loc[72:, [col + '_mult' for col in z_columns]].sum()

        # Align the indices by renaming the multiplied columns to match the originals
        column_sums_from_72.index = [col.replace('_mult', '') for col in column_sums_from_72.index]

        # Now perform the division
        column_ratios = column_sums_from_72 / column_sums

        # Add the column sums as a new row at the bottom of the DataFrame
        df_multiplied.loc['Column Sums'] = column_sums

        # Add the column_sums_from_72 as a new row at the bottom of the DataFrame
        df_multiplied.loc['Column Sums from 72 onward'] = column_sums_from_72

        # Add the column_ratios as a new row at the bottom of the DataFrame
        df_multiplied.loc['Column Ratios'] = column_ratios

        df_multiplied.to_csv(file, index=True)
        print(f' Saved as {file}')
        # return df_multiplied
