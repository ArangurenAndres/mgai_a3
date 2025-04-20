import matplotlib.pyplot as plt
import numpy as np
import os


import os
import numpy as np
import matplotlib.pyplot as plt

def read_image(filename: str = None) -> np.ndarray:
    if filename is None:
        raise ValueError("Filename must be provided.")

    if os.path.exists(filename):
        img_t = plt.imread(filename)
        print(f"Image loaded: {filename}")
        print(f"Dimensions: {img_t.shape}")  # (H, W, C)

        plt.figure(figsize=(img_t.shape[1] / 100, img_t.shape[0] / 100), dpi=100) # Remove border
        plt.imshow(img_t)
        plt.axis('off')
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        plt.show()

        return img_t
    else:
        print(f"File '{filename}' does not exist.")
        return None

def read_processed():
    pass



if __name__ == "__main__":
    img_name = "mario_1_1.png"
    img_path = os.path.join(os.path.dirname(__file__),'..','data','original', img_name)
    read_image(img_path)
    