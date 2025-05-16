import os
import torch
import numpy as np
from utils.process_data import ProcessDataSymbolic
from models.diffusion_model import build_diffusion_model_from_config, load_config

@torch.no_grad()
def sample_from_diffusion(model, schedule, shape, device):
    """
    Reverse diffusion process: x_T -> x_0
    """
    x_t = torch.randn(shape, device=device)

    for t in reversed(range(schedule.timesteps)):
        t_tensor = torch.full((shape[0],), t, device=device, dtype=torch.long)
        predicted_noise = model.unet(x_t, t_tensor)

        alpha = schedule.alphas[t]
        alpha_bar = schedule.alpha_bars[t]
        beta = schedule.betas[t]

        if t > 0:
            noise = torch.randn_like(x_t)
        else:
            noise = torch.zeros_like(x_t)

        # DDPM denoising step
        x_t = (1 / alpha.sqrt()) * (x_t - ((1 - alpha) / (1 - alpha_bar).sqrt()) * predicted_noise) + beta.sqrt() * noise

    return x_t

def generate():
    # Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "config.yaml")
    model_path = os.path.join(script_dir, "results", "model_final.pt")
    mapping_path = os.path.join(script_dir, "utils", "mapping.yaml")

    # Load config and model
    config = load_config(config_path)
    model, schedule = build_diffusion_model_from_config(config_path, device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Load processor for mapping back to symbols
    processor = ProcessDataSymbolic(config_path=config_path, mapping_path=mapping_path)

    # Get patch size and number of classes
    H, W = config["data_process"]["sliding_window"]
    C = config["model"]["in_channels"]
    shape = (1, C, H, W)

    print("🎲 Sampling a new level patch...")
    output = sample_from_diffusion(model, schedule, shape, device)  # shape: [1, C, H, W]

    # Convert to one-hot
    one_hot = torch.argmax(output.squeeze(0), dim=0).cpu().numpy()  # shape: [H, W]
    id_file = one_hot.tolist()

    # Convert back to symbolic
    symbolic_level = processor.convert_identity_to_symbolic(id_file)

    print("🧱 Generated symbolic patch:\n")
    for row in symbolic_level:
        print(row)

    # Optionally save to file
    output_path = os.path.join(script_dir, "results", "generated_patch.txt")
    with open(output_path, "w") as f:
        for row in symbolic_level:
            f.write(row + "\n")
    print(f"\n📄 Saved to: {output_path}")

if __name__ == "__main__":
    generate()
