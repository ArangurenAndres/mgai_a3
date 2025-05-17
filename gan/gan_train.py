import torch
import torch.nn as nn
import torch.optim as optim
import os
import sys
from torch.utils.data import Dataset, DataLoader
import time
import matplotlib.pyplot as plt

# Add parent directory to path for imports
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from levels.utils.process_data import ProcessDataSymbolic
from gan_model import MarioLevelGenerator, MarioLevelDiscriminator

class MarioLevelDataset(Dataset):
    def __init__(self, level_patches):
        # List of level patches in one-hot encoded format
        self.level_patches = [torch.tensor(patch, dtype=torch.float32) for patch in level_patches]
    
    def __len__(self):
        return len(self.level_patches) # Returns the number of level patches for one level
    
    def __getitem__(self, idx):
        return self.level_patches[idx] # Returns the level patch at the given index
    
def train_gan(generator, discriminator, dataloader, num_epochs=100, lr_g=0.0002, lr_d=0.0001, device='cpu'):
    generator.to(device)
    discriminator.to(device)
    
    # Define loss function and optimizers
    criterion = nn.BCELoss() # Binary Cross-Entropy Loss
    optimizer_g = optim.Adam(generator.parameters(), lr=lr_g, betas=(0.5, 0.999)) # Generator optimizer
    optimizer_d = optim.Adam(discriminator.parameters(), lr=lr_d, betas=(0.5, 0.999)) # Discriminator optimizer
    
    # Track losses
    g_losses = []
    d_losses = []
    
    print("Starting GAN training...")
    for epoch in range(num_epochs):
        total_g_loss = 0
        total_d_loss = 0
        
        for real_patches in dataloader:
            batch_size = real_patches.size(0)
            real_patches = real_patches.to(device)
            
            # Create labels for real and fake data
            real_labels = torch.ones(batch_size, 1, device=device)
            fake_labels = torch.zeros(batch_size, 1, device=device)
            
            # TRAIN DISCRIMINATOR
            discriminator.zero_grad()
            
            # Train on real patches
            real_outputs = discriminator(real_patches)
            d_loss_real = criterion(real_outputs, real_labels)
            
            # Train on fake patches
            z = torch.randn(batch_size, generator.latent_dim, device=device) # Random noise
            fake_patches = generator(z)
            fake_outputs = discriminator(fake_patches.detach()) # Detach to avoid training generator
            d_loss_fake = criterion(fake_outputs, fake_labels)
            
            # Total discriminator oss
            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            optimizer_d.step()
            
            # TRAIN GENERATOR
            generator.zero_grad()
            
            # Generate fake patches to try and fool the discriminator
            fake_outputs = discriminator(fake_patches)
            g_loss = criterion(fake_outputs, real_labels) # Generator tries to fools discriminator into thinking the generated patches are real
            g_loss.backward() # The results help us update parameters
            optimizer_g.step()
            
            # Add up losses
            total_d_loss += d_loss.item()
            total_g_loss += g_loss.item()
        
        # Avg losses for the epoch
        avg_d_loss = total_d_loss / len(dataloader)
        avg_g_loss = total_g_loss / len(dataloader)
        d_losses.append(avg_d_loss)
        g_losses.append(avg_g_loss)
        
        print(f"Epoch [{epoch+1}/{num_epochs}], D loss: {avg_d_loss:.4f}, G loss: {avg_g_loss:.4f}")
    
    print("GAN training completed!")
    return generator, discriminator, g_losses, d_losses

if __name__ == "__main__":
    # load and process data
    config_path = os.path.abspath(os.path.join(parent_dir, 'levels', 'config.yaml'))
    mapping_path = os.path.abspath(os.path.join(parent_dir, 'levels', 'utils', 'mapping.yaml'))
    
    processor = ProcessDataSymbolic(config_path=config_path, mapping_path=mapping_path)
    
    # Get symbolic data folder and files
    symb_data_folder = processor.folder_path
    symbolic_files = [f for f in os.listdir(symb_data_folder) if f.endswith('.txt')]
    
    if not symbolic_files:
        raise FileNotFoundError(f"No .txt files found in the symbolic data folder: {symb_data_folder}")
    
    # Collect all one-hot encoded patches
    all_one_hot_patches = []
    print(f"Loading and processing {len(symbolic_files)} symbolic files from {symb_data_folder}")
    for symb_file in symbolic_files:
        print(f"Processing {symb_file}...")
        try:
            processor.load_symbolic(symb_file)
            patches = processor.crop_symbolic()
            for patch in patches:
                _, vector_file = processor.forward_mapping(patch)
                all_one_hot_patches.append(vector_file)
        except Exception as e:
            print(f"Error processing {symb_file}: {e}")
            continue
    
    print(f"Finished processing all files. Total patches collected: {len(all_one_hot_patches)}")
    
    # Create dataset and dataloader
    if not all_one_hot_patches:
        raise ValueError("No valid patches found. Please check the symbolic files.")
    
    dataset = MarioLevelDataset(all_one_hot_patches)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Get level dimensions
    level_height, level_width = all_one_hot_patches[0].shape[:2]
    n_tile_types = all_one_hot_patches[0].shape[2]
    
    
    # Create generator and discriminator
    latent_dim = 32
    generator = MarioLevelGenerator(latent_dim=latent_dim, n_tile_types=10)
    discriminator = MarioLevelDiscriminator(latent_dim=latent_dim, n_tile_types=10)
    
    # Train GAN
    trained_generator, trained_discriminator, g_losses, d_losses = train_gan(generator, discriminator, dataloader)
    
    # Save the trained models
    model_save_dir = os.path.join(os.path.dirname(__file__), 'gan_models')
    os.makedirs(model_save_dir, exist_ok=True)
    torch.save(trained_generator.state_dict(), os.path.join(model_save_dir, 'gan_mario_generator.pth'))
    torch.save(trained_discriminator.state_dict(), os.path.join(model_save_dir, 'gan_mario_discriminator.pth'))
    print(f"Trained models saved to {model_save_dir}")
    
    # Plot the loss curves
    plt.figure(figsize=(10, 5))
    plt.plot(g_losses, label='Generator Loss')
    plt.plot(d_losses, label='Discriminator Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('GAN Training Loss')
    plt.legend()
    plt.grid(True)
    
    # Generate a sample level using the trained generator
    patch_width = trained_generator.patch_width
    num_patches = patch_width * 7
    
    print(f"Attempting to generate a whole level of width: {num_patches} tiles...")
    symbolic_level = trained_generator.generate_whole_level(num_patches, processor)
    
    # Create output directory for generated levels 
    output_dir = os.path.join(os.path.dirname(__file__), 'generated_levels_GAN')
    os.makedirs(output_dir, exist_ok=True)
    
    # Create filename with timestamp to avoid overwriting
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    output_file = os.path.join(output_dir, f'gan_level_{timestamp}.txt')
    
    # Also save the plot with the same timestamp
    plot_path = os.path.join(output_dir, f'training_loss_{timestamp}')
    plt.savefig(plot_path)
    print(f"Training loss plot saved to: {plot_path}")
    plt.close()
    
    # Save the generated level to a file
    print(f"Saving generated level to: {output_file}")
    with open(output_file, 'w') as f:
        for row in symbolic_level:
            f.write(row + '\n')
    
    # Display the generated level
    print("\nGenerated level after training:")
    processor.visualize_file(symbolic_level)
    
                
        
