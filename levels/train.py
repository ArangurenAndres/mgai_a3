import os
import json
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

# Local imports
from utils.process_data import ProcessDataSymbolic
from utils.dataloader import MarioLevelDataset
from models.diffusion_model import build_diffusion_model_from_config, load_config


def train():
    train_losses = []
    val_losses = []

    # -----------------------------
    # Setup
    # -----------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "config.yaml")
    mapping_path = os.path.join(script_dir, "utils", "mapping.yaml")
    results_dir = os.path.join(script_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    # Load config
    config = load_config(config_path)
    model_cfg = config["model"]
    train_cfg = config["train"]
    data_cfg = config["data_process"]

    # -----------------------------
    # Load symbolic data
    # -----------------------------
    processor = ProcessDataSymbolic(
        config_path=config_path,
        mapping_path=mapping_path
    )

    filename = data_cfg.get("filename", "mario_1_1.txt")
    processor.load_symbolic(filename)
    patches = processor.crop_symbolic()

    full_dataset = MarioLevelDataset(patches=patches, processor=processor)

    # -----------------------------
    # Train/val split (sequential)
    # -----------------------------
    val_ratio = train_cfg.get("val_split", 0.1)
    val_size = int(len(full_dataset) * val_ratio)
    train_size = len(full_dataset) - val_size

    train_indices = list(range(train_size))
    val_indices = list(range(train_size, len(full_dataset)))

    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)

    train_loader = DataLoader(train_dataset, batch_size=train_cfg["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=train_cfg["batch_size"], shuffle=False)

    # -----------------------------
    # Load model & optimizer
    # -----------------------------
    model, schedule = build_diffusion_model_from_config(config_path, device)
    optimizer = optim.Adam(model.parameters(), lr=train_cfg["learning_rate"])

    # -----------------------------
    # Training loop
    # -----------------------------
    print_every = train_cfg.get("print_every", 10)
    epochs = train_cfg["epochs"]

    print(f"🚀 Starting training for {epochs} epochs (Train: {train_size}, Val: {val_size})...\n")

    for epoch in tqdm(range(epochs), desc="Epochs"):
        model.train()
        running_loss = 0.0
        batch_losses = []

        batch_iter = tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False)
        for i, batch in enumerate(batch_iter):
            batch = batch.to(device)
            t = torch.randint(0, schedule.timesteps, (batch.size(0),), device=device).long()

            loss = model(batch, t)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            batch_losses.append(loss.item())
            batch_iter.set_postfix(loss=loss.item())

        avg_train_loss = sum(batch_losses) / len(batch_losses)
        train_losses.append(avg_train_loss)

        # -----------------------------
        # Validation loop
        # -----------------------------
        model.eval()
        val_loss_list = []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                t = torch.randint(0, schedule.timesteps, (batch.size(0),), device=device).long()
                loss = model(batch, t)
                val_loss_list.append(loss.item())

        avg_val_loss = sum(val_loss_list) / len(val_loss_list)
        val_losses.append(avg_val_loss)

        print(f"✅ Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    # -----------------------------
    # Save final model and loss logs
    # -----------------------------
    torch.save(model.state_dict(), os.path.join(results_dir, "model_final.pt"))

    loss_dict = {
        "train": train_losses,
        "val": val_losses
    }
    with open(os.path.join(results_dir, "losses.json"), "w") as f:
        json.dump(loss_dict, f, indent=2)

    print("\n✅ Saved model and loss logs to 'results/'")
    print("🎉 Training complete.")
    return train_losses, val_losses


if __name__ == "__main__":
    train_losses, val_losses = train()
