import os
import re

import cv2
import numpy as np
import pandas as pd
from cv2.typing import MatLike
from matplotlib import pyplot as plt
from matplotlib.figure import Figure


class RoiAnalyser:
    """
    This class provides functionality for cell cluster segmentation, contour detection,
    and histogram analysis of microscopy images, specifically DAPI, GFP,
    and TRITC channels.
    """

    def __init__(
        self,
        data_dir: str,
        roi_name: str,
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
            data_dir (str): Directory containing input images.
            out_dir (str, optional): Directory where results will be saved. Defaults to "out".
            plot (bool, optional): Whether to generate visualization plots. Defaults to True.
            size_threshold (int, optional): Minimum size for clusters in pixels. Defaults to 3000.
            opening_kernel (int, optional): Size of kernel for opening operations. Defaults to 13.
            opening_iter (int, optional): Number of iterations for opening. Defaults to 1.
            dilatation_kernel (int, optional): Size of kernel for dilation. Defaults to 7.
            dilatation_iter (int, optional): Number of iterations for dilation. Defaults to 2.
        """

        self.DATA_DIR = data_dir
        self.OUTPUT_DIR = os.path.join(out_dir, roi_name)
        self.PLOT = plot

        self.OPENING_KERNEL = opening_kernel
        self.OPENING_ITER = opening_iter
        self.DILATATION_ITER = dilatation_iter
        self.DILATATION_KERNEL = dilatation_kernel
        self.SIZE_THRESHOLD = size_threshold

        self.ROI_NAME = roi_name
        self.MODALITIES = ("dapi", "tritc", "gfp")
        self.z = ""

        os.makedirs(self.OUTPUT_DIR, exist_ok=True)

    def create_binary_mask(self, gray: MatLike) -> MatLike:
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

    def define_clusters(self, mask: MatLike) -> list[MatLike]:
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
    ) -> Figure:
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
            coords = list(contour[0][0])
            if coords[0] == 0:
                coords[0] += 100
            elif coords[0] >= dapi.shape[0] - 1:
                coords[0] -= 100
            
            if coords[1] == 0:
                coords[1] += 100
            elif coords[1] >= dapi.shape[1] - 1:
                coords[1] -= 100

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

        if self.PLOT:
            plt.show()
        else:
            plt.close()

        if self.z == "":
            fig.savefig(os.path.join(self.OUTPUT_DIR, f"{self.ROI_NAME}.png"))
        else:
            fig.savefig(os.path.join(self.OUTPUT_DIR, f"{self.ROI_NAME}_{self.z}.png"))
        return fig

    def build_histogram(
        self,
        channels: dict[str, MatLike],
        clusters: list[MatLike],
        channel_histograms: dict,
    ) -> dict:
        """
        Calculates pixel intensity histograms for each cluster.
        """

        num_bins = 256
        bin_edges = np.arange(num_bins + 1)

        # nested dictionary that stores histogram data for each channel
        # {'ChannelName': {'Z4' : {'Cluster_0': counts_array, 'Cluster_1': counts_array, ...}, {'Z6': {'Cluster_0': counts_array, 'Cluster_1': counts_array, ...}}}}
        
        z = self.z if self.z != "" else "Default"
        for name in channel_histograms.keys():
            channel_histograms[name][z] = {}

        # iterate over each cluster
        for i, cluster in enumerate(clusters):
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
                    column_name = f"Cluster {i + 1}"
                    channel_histograms[channel_name][z][column_name] = (
                        histogram_counts
                    )

        return channel_histograms

    def save_histogram(self, channel_histograms: dict) -> dict:
        """
        Saves computed histograms and *optionaly* figures.
        Returns dict with histograms mapped to channels.
        """

        print(f"\nSaving histogram data for ROI: {self.ROI_NAME}")

        out_dict = {}
        for channel_name, ch_data in channel_histograms.items():
            data_dict = {}
            for z_level, z_data in ch_data.items():
                data_dict[z_level] = pd.DataFrame(z_data)

            output_filename = os.path.join(
                self.OUTPUT_DIR, f"{self.ROI_NAME}_{channel_name}.csv"
            )

            final_df = pd.concat(data_dict, axis=1)
            pd.DataFrame.insert(
                final_df, loc=0, column=("", "Pixel values"), value=np.arange(256)
            )

            out_dict[channel_name] = output_filename
            # out_files.append(output_filename)
            final_df.to_csv(output_filename, index=False)
            print(
                f"  Successfully saved {channel_name} histogram data to {output_filename}"
            )

        return out_dict

    def apopnec_ratio(self, file: str, start_row: int):
        """
        Runs some analysis idk.
        """

        df = pd.read_csv(file, header=[0, 1], index_col=0)
        # print(df.head())

        df = df.apply(pd.to_numeric, errors="coerce")

        z_columns = df.columns
        # print(z_columns)

        column_sums = df[z_columns].sum()

        # 4. Multiply each column by the index ("Pixel values")
        df_multiplied = df.copy()
        for z_col in z_columns:
            new_col = (z_col[0], z_col[1] + "_mult")  # Add suffix to second level
            df_multiplied[new_col] = df[z_col] * df.index

        # Get multiplied columns
        z_columns_mult = [(z[0], z[1] + "_mult") for z in z_columns]

        # Sum original and multiplied from start_row onward
        column_sums = df[z_columns].sum()
        column_sums_from_start_row = df_multiplied.loc[start_row:, z_columns_mult].sum()

        # Rename the multiplied sum index to match the originals (remove "_mult")
        column_sums_from_start_row.index = pd.MultiIndex.from_tuples([
            (z[0], z[1].replace("_mult", "")) for z in column_sums_from_start_row.index
        ])

        # Align indices to perform division
        column_ratios = column_sums_from_start_row / column_sums

        # Step 2: Append the empty row first
        empty_row = pd.Series({col: pd.NA for col in df_multiplied.columns}, name="")
        df_multiplied = pd.concat([df_multiplied, pd.DataFrame([empty_row])])

        # Add summary rows to the DataFrame
        df_multiplied.loc["Column Sums"] = pd.Series(column_sums)
        df_multiplied.loc[f"Column Sums from {start_row} onward"] = pd.Series(
            column_sums_from_start_row
        )
        df_multiplied.loc["Column Ratios"] = pd.Series(column_ratios)

        # Save to file
        output_file = f"{file.split('.')[0]}.csv"
        df_multiplied.to_csv(output_file)
        print(f"Saved as {output_file}")

        return

    def get_file_paths(self) -> dict[str, str]:
        """ """

        file_names = os.listdir(self.DATA_DIR)
        file_paths = {modality: "" for modality in self.MODALITIES}

        # pattern = re.compile(
        #     rf"(?P<roi>{re.escape(self.ROI_NAME)})[-_\d]*.*?(?P<z>{re.escape(self.z)}).*?\b(?P<modality>DAPI|GFP|TRITC)"
        # )

        print("Loaded files:")
        for file_name in file_names:
            if self.ROI_NAME in file_name and self.z in file_name:
                for modality in file_paths.keys():
                    if modality.upper() in file_name:
                        file_paths[modality] = file_name
                        print(f"{modality}: {file_name}")

        return file_paths

    def load_data(
        self, dapi_path: str, gfp_path: str, tritc_path: str
    ) -> dict[str, dict[str, MatLike]]:
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
        self, z: str, clusters: list[MatLike], channel_histograms: dict
    ) -> dict:
        """
        Repeat the analysis for additional images.
        """

        print(" Repeating for: ", z)

        self.z = z
        file_paths = self.get_file_paths()
        imgs = self.load_data(
            dapi_path=file_paths["dapi"],
            gfp_path=file_paths["gfp"],
            tritc_path=file_paths["tritc"],
        )

        self.plot_contours(
            imgs["dapi"]["img"], imgs["gfp"]["img"], imgs["tritc"]["img"], clusters
        )

        channels = {"GFP": imgs["gfp"]["gray"], "TRITC": imgs["tritc"]["gray"]}
        channel_histograms = self.build_histogram(
            channels, clusters, channel_histograms
        )

        return channel_histograms

    def run_analysis(self, z="") -> tuple[list[MatLike], dict]:
        """
        Runs analysis for the first image group (DAPI, GFP and TRITC).
        """

        self.z = z
        file_paths = self.get_file_paths()
        imgs = self.load_data(
            dapi_path=file_paths["dapi"],
            gfp_path=file_paths["gfp"],
            tritc_path=file_paths["tritc"],
        )

        print(f"\nAnalysing: {self.ROI_NAME}, {self.z}")

        print("Step 1: Preprocessing DAPI image...")
        mask = self.create_binary_mask(imgs["dapi"]["gray"])

        print("Step 2: Finding contours...")
        clusters = self.define_clusters(mask)

        # figure = None
        figure = self.plot_contours(
            imgs["dapi"]["img"], imgs["gfp"]["img"], imgs["tritc"]["img"], clusters
        )

        print("Step 3: Building GFP and TRITC histograms for clusters...")
        channels = {"GFP": imgs["gfp"]["gray"], "TRITC": imgs["tritc"]["gray"]}
        channel_histograms = {"GFP": {}, "TRITC": {}}

        channel_histograms = self.build_histogram(
            channels, clusters, channel_histograms
        )

        return clusters, channel_histograms


if __name__ == "__main__":
    DATA_DIR = "data/#2451333014_ZProj_B IVA76"  # input data directory path
    OUTPUT_DIR = "out"  # output results directory path
    PLOT = False  # whether to show plots
    SIZE_THRESHOLD = 3000  # size threshold for filtering clusters

    roi_name = "E6ROI5"
    analyser = RoiAnalyser(DATA_DIR, roi_name, OUTPUT_DIR, PLOT, SIZE_THRESHOLD)

    # For dapi
    # dapi_path = "A1ROI1_02_2_1Z4_Confocal DAPI_001.tif"  # change me
    clusters, channel_histograms = analyser.run_analysis(z="")

    # For other Zs
    z = "Z4"  # change me
    print("Step 4: Repeat for additional Zs...")
    # channels = analyser.repeat_for_additional_images(z, clusters, channel_histograms)

    # Saving and exporting
    # save histograms
    print("Step 5: Saving histograms...")
    out_files = analyser.save_histogram(channel_histograms)
    print(out_files)

    # change these
    params = {
        "GFP": {"file": out_files["GFP"], "start_row": 72},
        "TRITC": {"file": out_files["TRITC"], "start_row": 72},
    }

    # Doing some analysis
    print("Step 6: Calculate apopnec ratio")
    for name, args in params.items():
        analyser.apopnec_ratio(file=args["file"], start_row=args["start_row"])
        print(f"|Done with {name}|\n")

    print("------------------")
    print("Done :)")
