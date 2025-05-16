import os
import json
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def plot_train_val_loss():
    # Resolve path relative to the script's location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(script_dir, "results", "losses.json")
    save_path = os.path.join(script_dir, "results", "loss_curve.png")

    # Load loss data
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Could not find: {json_path}")
        
    with open(json_path, "r") as f:
        losses = json.load(f)

    train_losses = losses["train"]
    val_losses = losses["val"]
    epochs = list(range(1, len(train_losses) + 1))

    # Prepare data for seaborn
    data = {
        "Epoch": epochs + epochs,
        "Loss": train_losses + val_losses,
        "Type": ["Train"] * len(train_losses) + ["Validation"] * len(val_losses)
    }

    df = pd.DataFrame(data)

    # Plot
    sns.set(style="whitegrid")
    plt.figure(figsize=(8, 5))
    sns.lineplot(data=df, x="Epoch", y="Loss", hue="Type", marker="o")
    plt.title("Training vs. Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.tight_layout()
    plt.savefig(save_path)
    print(f" Loss curve saved to: {save_path}")
    plt.show()


if __name__ == "__main__":
    plot_train_val_loss()
