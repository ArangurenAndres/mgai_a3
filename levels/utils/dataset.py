from load_files import ProcessDataSymbolic
import os
import numpy as np

class SymbolicPatchDataset:
    def __init__(self, config_path, mapping_path, data_folder):
        self.processor = ProcessDataSymbolic(config_path, mapping_path)
        self.data_folder = data_folder
        self.all_patches = []

        self._load_all()

    def _load_all(self):
        for fname in os.listdir(self.data_folder):
            if fname.endswith('.txt'):
                print(f"Processing {fname}...")
                self.processor.load_symbolic(fname)
                patches = self.processor.crop_symbolic()
                for patch in patches:
                    id_file, vec = self.processor.forward_mapping(patch)
                    self.all_patches.append(vec)  # or store (vec, id_file) for supervised tasks

    def get_data(self):
        return self.all_patches

if __name__ == "__main__":
    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'config.yaml'))
    mapping_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'mapping.yaml'))
    symb_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'symbol'))


    dataset = SymbolicPatchDataset(config_path, mapping_path, symb_folder) # Initialize the dataset
    patches = dataset.get_data()  # list of (H, W, C) numpy arrays

    # Convert to a single numpy tensor if needed:
    X = np.stack(patches)
    print("Final dataset shape:", X.shape)