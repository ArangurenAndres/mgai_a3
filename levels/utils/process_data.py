import matplotlib.pyplot as plt
import numpy as np
import os
import math
from load_files import load_config


class ProcessData:
    def __init__(self, config_path: str = None, img_name: str = None):
        self.config = load_config(config_path)
        self.folder_path = self.config["data_process"]["folder_path"]
        self.window_dim = self.config["data_process"]["sliding_window"]  # (tiles_H, tiles_W)
        self.stride = self.config["data_process"].get("stride", 1)       # in pixels
        self.img_name = img_name
        self.image = None
        self.tile_scale = None

    def read_image_data(self, show: bool = False) -> tuple[np.ndarray, int]:
        if not os.path.exists(self.folder_path):
            raise FileNotFoundError(f"File not found: {self.folder_path}")
        
        img_path = os.path.join(self.folder_path, self.img_name)
        if not img_path.endswith('.png'):
            raise ValueError("The file is not a PNG image")

        img_t = plt.imread(img_path)
        img_h, img_w, img_c = img_t.shape
        win_h, _ = self.window_dim
        tile_scale = math.ceil(img_h / win_h)

        print(f"Loaded image with dimensions H, W, C: {img_h}, {img_w}, {img_c}")
        print(f"Tile scale: {tile_scale} pixels")

        if show:
            plt.figure(figsize=(img_t.shape[1] / 100, img_t.shape[0] / 100), dpi=100)
            plt.imshow(img_t)
            plt.axis('off')
            plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
            plt.show()

        self.image = img_t
        self.tile_scale = tile_scale
        return img_t, tile_scale

    def crop_horizontal_strips(self) -> list:
        """
        Slide a window horizontally using stride (in tiles) from config and crop patches of full height.

        Returns:
            List of image patches (each a NumPy array).
        """
        if self.image is None or self.tile_scale is None:
            raise ValueError("Call read_image_data() before cropping.")

        img_h, img_w, _ = self.image.shape
        win_w_px = self.window_dim[1] * self.tile_scale  # width in pixels
        stride_px = self.stride * self.tile_scale        # stride in pixels

        patches = []

        for x in range(0, img_w - win_w_px + 1, stride_px):
            patch = self.image[:, x:x + win_w_px, :]
            patches.append(patch)

        print(f"Extracted {len(patches)} patches.")
        return patches



if __name__ == "__main__":
    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'config.yaml'))
    processor = ProcessData(config_path=config_path, img_name="mario_1_1.png")
    processor.read_image_data(show=False)
    patches = processor.crop_horizontal_strips()
    print("Displaying first 10 patches...")
    samples = 15
    for i, patch in enumerate(patches[:samples]):
        plt.imshow(patch)
        plt.axis('off')
        plt.title(f"Patch {i+1}")
        plt.show()

