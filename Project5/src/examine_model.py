# John Doe — Analyzes the trained MNIST CNN by visualizing learned weights and filters

# import statements
import sys
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Class Definitions ────────────────────────────────────────────────────────


class MNISTNetwork(nn.Module):
    """Same CNN architecture as mnist_cnn.py — must match exactly to load saved weights."""

    def __init__(self):
        super(MNISTNetwork, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=10, out_channels=20, kernel_size=5)
        self.dropout = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    # Computes a forward pass; returns log_softmax probabilities over 10 digit classes
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), kernel_size=2))
        x = F.relu(F.max_pool2d(self.dropout(self.conv2(x)), kernel_size=2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        return F.log_softmax(self.fc2(x), dim=1)


# ── Useful Functions ──────────────────────────────────────────────────────────


def load_model(path, device):
    """Loads the trained MNISTNetwork from a saved state dict file."""
    model = MNISTNetwork().to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    print(f"Model loaded from '{path}'\n")
    return model


def plot_filter_outputs(model, save_path="plot/filter_outputs.png"):
    """Passes the first MNIST test image through conv1 and plots each filter
    side by side with its corresponding output image — 5 rows x 4 columns
    (filter | output | filter | output)."""
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader

    # Load a single test image from MNIST
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    test_set = datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)
    image, label = next(iter(test_loader))  # shape: [1, 1, 28, 28]

    with torch.no_grad():
        # Move image to same device as the model before passing through conv1
        # output shape: [1, 10, 24, 24]  (10 filters, 28-5+1=24)
        output = model.conv1(image.to(next(model.parameters()).device))

    weights = model.conv1.weight.data.cpu()  # shape: [10, 1, 5, 5]

    # Layout: 5 rows × 4 cols — each row shows two (filter, output) pairs
    fig, axes = plt.subplots(5, 4, figsize=(10, 12))
    fig.suptitle(f"Conv1 filters and outputs for digit '{label.item()}'", fontsize=13)

    for i in range(10):
        row = i // 2  # 0-4
        col = (i % 2) * 2  # 0 or 2 — filter in even col, output in odd col

        # --- Filter (5x5) ---
        filt = weights[i].squeeze()
        filt = (filt - filt.min()) / (filt.max() - filt.min())
        axes[row, col].imshow(filt, cmap="gray")
        axes[row, col].set_title(f"Filter {i}", fontsize=9)
        axes[row, col].axis("off")

        # --- Filter output (24x24) ---
        filt_out = output[0, i].cpu()
        filt_out = (filt_out - filt_out.min()) / (filt_out.max() - filt_out.min())
        axes[row, col + 1].imshow(filt_out, cmap="gray")
        axes[row, col + 1].set_title(f"Output {i}", fontsize=9)
        axes[row, col + 1].axis("off")

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Filter output plot saved to '{save_path}'")
    plt.show()


def plot_first_layer_filters(model, save_path="plot/first_layer_filters.png"):
    """Extracts and plots the 10 learned 5x5 filters from the first conv layer."""
    # weights shape: [out_channels, in_channels, H, W] = [10, 1, 5, 5]
    weights = model.conv1.weight.data.cpu()

    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    fig.suptitle("Conv1 — 10 Learned Filters (5×5)", fontsize=13)

    for i, ax in enumerate(axes.flat):
        # squeeze out the single input channel → shape [5, 5]
        filt = weights[i].squeeze()

        # normalize each filter to [0, 1] for clean visualization
        filt = (filt - filt.min()) / (filt.max() - filt.min())

        ax.imshow(filt, cmap="gray")
        ax.set_title(f"Filter {i}", fontsize=9)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Filter plot saved to '{save_path}'")
    plt.show()


# ── Main Function ─────────────────────────────────────────────────────────────


def main(argv):
    # Usage: python3 analyze_model.py [model_path]
    # Example: python3 analyze_model.py mnist_cnn.pth
    model_path = argv[1] if len(argv) > 1 else "model/mnist_cnn.pth"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load the trained model
    model = load_model(model_path, device)

    # Visualize the first layer filters
    plot_first_layer_filters(model)

    # Visualize the effect of the 10 filters on the first test image
    plot_filter_outputs(model)

    return


if __name__ == "__main__":
    main(sys.argv)
