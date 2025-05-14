import torch
import torch.nn as nn
import numpy as np
import os
import sys

# Add parent directory to path for imports
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from levels.utils.process_data import ProcessDataSymbolic


class MarioLevelGenerator(nn.Module):
    def __init__(self, level_height, level_width, n_tile_types=10, latent_dim=32, hidden_dim=256):
        """
        MLP-based Mario level generator.
        
        Args:
            level_height (int): Height of level in tiles
            level_width (int): Width of level in tiles
            n_tile_types (int): Number of different tile types (default: 10)
            latent_dim (int): Size of the input noise vector (z)
            hidden_dim (int): Size of the hidden layers
        """
        super(MarioLevelGenerator, self).__init__()
        
        self.level_height = level_height
        self.level_width = level_width
        self.n_tile_types = n_tile_types
        self.latent_dim = latent_dim
        
        # Calculate output size: height × width × n_tile_types
        output_size = level_height * level_width * n_tile_types
        
        # Define the generator: A simple MLP feed-forward network that transforms the input noise (latent vector z) into level data
        self.main = nn.Sequential(
            # Input: latent vector z
            nn.Linear(latent_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Linear(hidden_dim * 2, hidden_dim * 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Output layer needs to be reshaped to (batch_size, height, width, n_tile_types)
            nn.Linear(hidden_dim * 4, output_size),
            nn.Sigmoid()  # Use sigmoid for probability distribution
        )
    
    def forward(self, z):
        """
        Forward pass
        
        Args:
            z (Tensor): Random noise vector of shape (batch_size, latent_dim)
            
        Returns:
            Tensor: Generated level of shape (batch_size, height, width, n_tile_types)
        """
        # Pass through network and reshape
        output = self.main(z)
        return output.view(-1, self.level_height, self.level_width, self.n_tile_types)
    
    def generate_level(self, batch_size=1, device='cpu'):
        """
        Generate a random Mario level
        
        Args:
            batch_size (int): Number of levels to generate
            device (str): Device to run the model on
            
        Returns:
            numpy.ndarray: Generated level(s) as one-hot encoded tensor
        """
        # Generate random noise
        z = torch.randn(batch_size, self.latent_dim, device=device)
        
        # Generate level
        with torch.no_grad():
            level = self(z)
            
        # Convert to numpy for processing with existing code
        return level.cpu().numpy()
    
    def generate_symbolic_level(self, processor, batch_size=1, device='cpu'):
        """
        Generate a symbolic Mario level
        
        Args:
            processor (ProcessDataSymbolic): Level processor for conversion
            batch_size (int): Number of levels to generate
            device (str): Device to run the model on
            
        Returns:
            list: Generated level in symbolic format (list of strings)
        """
        # Generate level as one-hot encoding
        vector_level = self.generate_level(batch_size, device)[0]  # Get first level
        
        # Convert to identity representation
        id_level = processor.convert_vector_to_id(vector_level)
        
        # Convert to symbolic representation
        symbolic_level = processor.convert_identity_to_symbolic(id_level)
        
        return symbolic_level

if __name__ == "__main__":
    # Example usage
    level_height = 14  # Example - adjust based on your data
    level_width = 28   # Example - adjust based on your data
    
    # Create model
    generator = MarioLevelGenerator(level_height, level_width)
    
    # Generate a random level
    level_tensor = generator.generate_level()
    print(f"Generated level shape: {level_tensor.shape}")
    
    # To convert to symbolic representation
    config_path = os.path.abspath(os.path.join(parent_dir, 'levels', 'config.yaml'))
    mapping_path = os.path.abspath(os.path.join(parent_dir, 'levels', 'utils', 'mapping.yaml'))
    
    processor = ProcessDataSymbolic(config_path=config_path, mapping_path=mapping_path)
    symbolic_level = generator.generate_symbolic_level(processor)
    
    print("Generated level:")
    processor.visualize_file(symbolic_level)