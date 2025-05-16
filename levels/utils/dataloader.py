import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

from .process_data import ProcessDataSymbolic  # assumes this file is called process_data.py


class MarioLevelDataset(Dataset):
    def __init__(self, patches, processor, num_classes=10):
        """
        Args:
            patches (list): symbolic patches from processor.crop_symbolic()
            processor (ProcessDataSymbolic): processor object for forward mapping
            num_classes (int): number of tile types
        """
        self.processor = processor
        self.patches = patches
        self.num_classes = num_classes
        self.data = self._build_dataset()

    def _build_dataset(self):
        all_tensors = []
        for patch in self.patches:
            _, vector = self.processor.forward_mapping(patch)  # shape: (H, W, C)
            vector = vector.astype('float32')
            vector = np.transpose(vector, (2, 0, 1))  # (C, H, W)
            all_tensors.append(torch.tensor(vector))
        return all_tensors

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


if __name__ == "__main__":
    # Config paths (adjust if needed)
    root_dir = os.path.dirname(__file__)
    config_path = os.path.join(root_dir, '..', 'config.yaml')
    mapping_path = os.path.join(root_dir, 'mapping.yaml')
    symb_file = 'mario_1_1.txt'
    n_patches = 5

    # Load symbolic patches
    processor = ProcessDataSymbolic(config_path=config_path, mapping_path=mapping_path)
    processor.load_symbolic(symb_file)
    patches = processor.crop_symbolic()
    print(f"Dataset length: {len(patches)}")

    # Build dataset
    dataset = MarioLevelDataset(patches=patches[:n_patches], processor=processor)
    loader = DataLoader(dataset, batch_size=2, shuffle=True)

    # Test loading
    for batch_idx, batch in enumerate(loader):
        print(f"Batch {batch_idx}")
        print(f"Shape: {batch.shape}")  # Should be (B, C, H, W)
        print(f"Data (first sample, channel 0):\n{batch[0][0]}")
        if batch_idx == 1:
            break
