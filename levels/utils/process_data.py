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

class ProcessDataSymbolic:
    def __init__(self, config_path: str = None, img_name: str = None):
        self.config = load_config(config_path)
        self.folder_path = self.config["data_process"]["folder_path"]
        self.window_dim = self.config["data_process"]["sliding_window"]  # (tiles_H, tiles_W)
        self.stride = self.config["data_process"].get("stride", 1)       # in pixels
        self.img_name = img_name
    
    def load_symbolic(self):
        # Presevers the exact format of the txt file including:
        # The number of lines (rows)
        # Number of characters per line (columns)
        
        if not os.path.exists(self.folder_path):
            raise FileNotFoundError(f"File not found: {self.folder_path}")
        img_path = os.path.join(self.folder_path,self.img_name)
        with open(img_path,'r') as f:
            lines = [line.rstrip('\n') for line in f]
        
        self.lines = lines
        
        #print the level
        for row in lines:
            print(row)
        return lines
    
    def crop_symbolic(self):
        win_h, win_w = self.window_dim
        #Get full dimensions of the level
        file_height = len(self.lines)
        file_width = len(self.lines[0])
        if file_height < win_h:
            raise ValueError(f"File height {file_height} is less than window height {win_h}")
        
        # Slide window horizontally
        patches = [] # store patches in a list
        for x in range(0,file_width-win_w,self.config["data_process"]["stride"]):
            patch = [line[x:x+win_w] for line in self.lines[:win_h]] #apply the cropping for the full height of the file
            patches.append(patch)
        print(f"Loaded level of size {file_height}x{file_width} characters")
        print(f"Extracted {len(patches)} patches of size {win_h}x{win_w}")
        self.patches = patches

        return patches
    def visualize_patches(self, n:int = 10):
        # Visualize the patches
        for i, patch in enumerate(self.patches[0:n]):
            print(f"Patch {i+1}:")
            for line in patch:
                print(line)
            print("\n")

    




if __name__ == "__main__":
    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'config.yaml'))
    symb_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'symbol'))
    symb_file = 'mario_1_1.txt' # File name
    txt_path = os.path.join(symb_folder, symb_file)
    processor = ProcessDataSymbolic(config_path=config_path, img_name="mario_1_1.txt") # Intialize processor class
    symb_file = processor.load_symbolic() #Load the symbolic data
    patches = processor.crop_symbolic() #Crop the symbolic data
    processor.visualize_patches(10) # Visualize the first n patches

