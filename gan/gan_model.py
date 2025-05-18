import torch
import torch.nn as nn
import numpy as np
import os 
import sys

# Add parent directory to path for imports
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
    
from levels.utils.process_data import ProcessDataSymbolic

class MarioLevelGenerator(nn.Module):
    def __init__(self, patch_height, patch_width, n_tile_types=10, latent_dim=100, hidden_dim=512):
        super(MarioLevelGenerator, self).__init__()
        
        self.patch_height = patch_height
        self.patch_width = patch_width
        self.n_tile_types = n_tile_types
        self.latent_dim = latent_dim
        
        # Calculate the output size
        output_size = patch_height * patch_width * n_tile_types
        
        # Generator network (more layers than MLP for more complex feature learning)
        self.main = nn.Sequential(
            # Input: latent vector z
            nn.Linear(latent_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(hidden_dim),
            
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(hidden_dim * 2),
            
            nn.Linear(hidden_dim * 2, hidden_dim * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(hidden_dim * 4),
            
            nn.Linear(hidden_dim * 4, output_size),
            nn.Sigmoid()
        )
    
    def forward(self, z):
        # Generate level
        output = self.main(z)
        return output.view(-1, self.patch_height, self.patch_width, self.n_tile_types)
    
    def generate_patch(self, batch_size=1, device='cpu'):
        # Generate random noise
        z = torch.randn(batch_size, self.latent_dim, device=device)
        
        # Switch to eval mode to avoid BatchNorm issues
        self.eval()
        
        # Generate patch
        with torch.no_grad():
            patch = self(z)
        
        # Switch back to training mode
        self.train()
        
        return patch.cpu().numpy()
    
    def generate_symbolic_patch(self, processor, batch_size=1, device='cpu'):
        # Generate level as one-hot encoding
        vector_patch = self.generate_patch(batch_size, device)[0]  # Get first level
        
        # Convert to identity representation
        id_patch = processor.convert_vector_to_id(vector_patch)
        
        # Convert to symbolic representation
        symbolic_patch = processor.convert_identity_to_symbolic(id_patch)
        
        return symbolic_patch
    
    def generate_whole_level(self, level_tile_width, processor, device='cpu'):
        # Generates a whole level by stitching together patches (might have seams between incoherent patches)
        patch_width = self.patch_width
        
        # Level width must be a multiple of patch width in order to stitch together multiple patches
        if level_tile_width % patch_width != 0:
            raise ValueError(f"Desired level width ({level_tile_width}) must be a multiple of patch width ({patch_width})")
        
        # Number of patches to stitch to generate one level
        num_patches = level_tile_width // patch_width
        
        # Generate enough patches to cover the desired width level_tile_width
        generated_patches = []
        print(f"Generating {num_patches} patches to stitch into one level...")
        for i in range(num_patches):
            # Generate patch
            print(f"Generating patch {i+1}/{num_patches}...")
            patch_vector = self.generate_patch(batch_size=1, device=device)[0] # Get the single patch
            generated_patches.append(patch_vector)
        
        # Stitch the patches horizontally
        print("Stitching patches horizontally...")
        
        # Initialize the full level array using the first patch
        full_level = np.concatenate(generated_patches, axis=1)
        
        # Convert the whole level vector back to symbolic format
        print("Converting stitched level to symbolic format...")
        id_level = processor.convert_vector_to_id(full_level)
        symbolic_level = processor.convert_identity_to_symbolic(id_level)
        
        print("Whole level generation complete.")
        return symbolic_level

class MarioLevelDiscriminator(nn.Module):
    def __init__(self, patch_height, patch_width, n_tile_types=10, hidden_dim=256):
        super(MarioLevelDiscriminator, self).__init__()
        
        # Calculate input size
        input_size = patch_height * patch_width * n_tile_types
        
        # Discriminator network
        self.main = nn.Sequential(
            # Input: flattened level patch
            nn.Linear(input_size, hidden_dim * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            
            # Output: single value (real/fake probability)
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, level):
        # Flatten input
        flattened_level = level.view(level.size(0), -1)
        
        # Return probability of the level being real)
        return self.main(flattened_level)
    
