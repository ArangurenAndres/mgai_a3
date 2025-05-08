import matplotlib.pyplot as plt
import numpy as np
import os
import math
from load_files import load_config, load_mapping


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
    def __init__(self, config_path: str = None, mapping_path: str = None):
        self.config = load_config(config_path)
        self.mapping = load_mapping(mapping_path)
        self.folder_path = self.config["data_process"]["folder_path"]
        self.window_dim = self.config["data_process"]["sliding_window"]  # (tiles_H, tiles_W)
        self.stride = self.config["data_process"].get("stride", 1)       # in pixels


    def load_symbolic(self,img_name: str = None):
        # Presevers the exact format of the txt file including:
        # The number of lines (rows)
        # Number of characters per line (columns)
        
        if not os.path.exists(self.folder_path):
            raise FileNotFoundError(f"File not found: {self.folder_path}")
        img_path = os.path.join(self.folder_path,img_name)
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
    # Forward mapping functions
    def convert_to_identity(self, symb_file = None, visualize: bool = False):
        if not hasattr(self, 'lines') or symb_file is None:
            raise ValueError("Symbolic data must be loaded before conversion.")
        
        if self.mapping.get("symbol_identity") is None:  # check if mapping file is loaded
            raise ValueError("Mapping data must be loaded before conversion.")
        symbol_to_id = self.mapping["symbol_identity"] # Load conversion mapping
        id_file = [
        [symbol_to_id.get(char, -1) for char in row]
        for row in symb_file
        ] # Using list comprehension covert the symbolic file to identity file
        return id_file
    
    def convert_id_to_vector(self, id_file: list, n_classes: int = 10) -> np.ndarray:
        id_array = np.array(id_file)
        h, w = id_array.shape

        # One-hot encode using numpy indexing
        one_hot = np.zeros((h, w, n_classes), dtype=np.uint8)
        for class_id in range(n_classes):
            one_hot[:, :, class_id] = (id_array == class_id)

        return one_hot
    #--------------------------
    # Backward mapping functions
    
    def convert_vector_to_id(self, vector_file: np.ndarray) -> list:
        id_file = np.argmax(vector_file, axis=-1) # Convert one-hot encoded vector back to identity
        return id_file.tolist()
    
    def convert_identity_to_symbolic(self, id_file: list) -> list:
        """
        Converts a 2D identity grid to symbolic characters using inverse mapping.
        """
        # Reverse the mapping: {0: 'X', 1: 'S', ...}
        id_to_symbol = {v: k for k, v in self.mapping["symbol_identity"].items()}
        
        symbolic_file = [
            ''.join(id_to_symbol.get(cell, '?') for cell in row)
            for row in id_file
        ]
        
        return symbolic_file
    # Forward mapping symbolic -> identify -> one-hot vector
    def forward_mapping(self, symb_file: list) -> list:
        # 1. Convert symbolic file to identify
        id_file = self.convert_to_identity(symb_file)
        # 2. Convert identify file to one-hot vector
        vector_file = self.convert_id_to_vector(id_file)
        return id_file,vector_file
    # Backward mapping one-hot vector -> identify -> symbolic
    def backward_mapping(self, vector_file: np.ndarray, orig_symb_file=None, orig_id_file=None) -> tuple:
        # 1. Convert vector to identity
        id_file = self.convert_vector_to_id(vector_file)

        if id_file != orig_id_file:
            raise ValueError("Identity file mismatch after decoding from vector. Aborting.")

        print("Identity file matches original.")

        # 2. Convert identity to symbolic
        symb_file = self.convert_identity_to_symbolic(id_file)

        if orig_symb_file is not None:
            if symb_file != orig_symb_file:
                raise ValueError("Symbolic file mismatch after reconstruction. Aborting.")

            print("Symbolic file matches original.")

        return id_file, symb_file


    @staticmethod
    def visualize_file(file=None):
        for line in file:
            print(line)
        print("\n")
    




if __name__ == "__main__":
    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'config.yaml'))
    mapping_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'mapping.yaml'))
    symb_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'symbol'))

    symb_file = 'mario_1_1.txt' # File name

    processor = ProcessDataSymbolic(config_path=config_path, mapping_path=mapping_path) # Intialize processor class
    symb_file = processor.load_symbolic(symb_file) #Load the symbolic data
    patches = processor.crop_symbolic() #Crop the symbolic data obtaining patches

    for id,patch in enumerate(patches[:1]):
        # Appply forward mapping
        id_file,vector_file = processor.forward_mapping(patch)
        print("Original symbolic file:")
        processor.visualize_file(patch)
        # Apply bacward mapping
        id_file_re, symb_file_re = processor.backward_mapping(vector_file, orig_symb_file=patch, orig_id_file=id_file)
        print("Reconstructed symbolic file:")
        processor.visualize_file(symb_file_re)


