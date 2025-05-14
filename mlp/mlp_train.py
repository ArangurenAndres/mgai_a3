import torch
import torch.nn as nn
import torch.optim as optim
import os
import sys
import numpy as np
from torch.utils.data import Dataset, DataLoader

# Add parent directory to path for imports
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from levels.utils.process_data import ProcessDataSymbolic
from mlp_model import MarioLevelGenerator


class MarioLevelDataset(Dataset):
    """Dataset for Mario levels"""
    
    def __init__(self, level_patches):
        """
        Args:
            level_patches (list): List of level patches in one-hot encoded format
        """
        self.level_patches = [torch.tensor(patch, dtype=torch.float32) for patch in level_patches]
    
    def __len__(self):
        return len(self.level_patches)
    
    def __getitem__(self, idx):
        return self.level_patches[idx]


def train_generator(generator, dataloader, num_epochs=100, lr=0.0002, device='cpu'):
    """
    Train the generator using binary cross entropy loss
    
    Args:
        generator (MarioLevelGenerator): Generator model
        dataloader (DataLoader): DataLoader for training data
        num_epochs (int): Number of training epochs
        lr (float): Learning rate
        device (str): Device to train on
    """
    generator.to(device)
    
    # Define loss and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    
    # Training loop
    for epoch in range(num_epochs):
        total_loss = 0
        
        for i, real_levels in enumerate(dataloader):
            batch_size = real_levels.size(0)
            real_levels = real_levels.to(device)
            
            # Generate random noise
            z = torch.randn(batch_size, generator.latent_dim, device=device)
            
            # Generate fake levels
            fake_levels = generator(z)
            
            # Train generator to produce levels similar to real ones
            optimizer.zero_grad()
            loss = criterion(fake_levels, real_levels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # Print training progress
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(dataloader):.4f}")
    
    print("Training completed!")
    return generator


if __name__ == "__main__":
    # Load and process data
    config_path = os.path.abspath(os.path.join(parent_dir, 'levels', 'config.yaml'))
    mapping_path = os.path.abspath(os.path.join(parent_dir, 'levels', 'utils', 'mapping.yaml'))
    
    processor = ProcessDataSymbolic(config_path=config_path, mapping_path=mapping_path)
    
    symb_data_folder = processor.folder_path
    symbolic_files = [f for f in os.listdir(symb_data_folder) if f.endswith('.txt')]
    
    if not symbolic_files:
        raise FileNotFoundError(f"No .txt files found in the symbolic data folder: {symb_data_folder}")
    
    all_one_hot_patches = []
    
    print(f"Loading and processing {len(symbolic_files)} symbolic files from {symb_data_folder}")
    for symb_file in symbolic_files:
        print(f"Processing {symb_file}...")
        
        try:
            # Load symbolic level data
            processor.load_symbolic(symb_file)
            
            # Crop level into patches
            patches = processor.crop_symbolic()

            # Convert patches to one-hot encoded vectors and collect them
            for patch in patches:
                id_file, vector_file = processor.forward_mapping(patch)
                all_one_hot_patches.append(vector_file)
        except Exception as e:
            print(f"Error processing {symb_file}: {e}")
            continue
    
    print(f"Finished processing all files. Total patches collected: {len(all_one_hot_patches)}")
    
    # Create dataset and dataloader from ALL collected patches
    if not all_one_hot_patches:
        raise ValueError("No valid patches found. Please check the symbolic files.")
    
    dataset = MarioLevelDataset(all_one_hot_patches)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Get level dimensions
    level_height, level_width = all_one_hot_patches[0].shape[:2]
    n_tile_types = all_one_hot_patches[0].shape[2]
    
    # Create and train generator
    generator = MarioLevelGenerator(level_height, level_width, n_tile_types)
    trained_generator = train_generator(generator, dataloader)
    
    # Save the trained model
    torch.save(trained_generator.state_dict(), 'mario_generator.pth')
    
    # Generate a sample level
    symbolic_level = trained_generator.generate_symbolic_level(processor)
    print("Generated level after training:")
    processor.visualize_file(symbolic_level)