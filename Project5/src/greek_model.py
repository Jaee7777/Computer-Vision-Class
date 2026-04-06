# John Doe — Transfer learning from MNIST CNN to recognize Greek letters α, β, γ

# import statements
import sys
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader


# ── Class Definitions ────────────────────────────────────────────────────────


class MNISTNetwork(nn.Module):
    """Original MNIST CNN — loaded to extract pretrained conv layer weights."""

    def __init__(self):
        super(MNISTNetwork, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=10, out_channels=20, kernel_size=5)
        self.dropout = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    # Computes a forward pass through the original MNIST network
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), kernel_size=2))
        x = F.relu(F.max_pool2d(self.dropout(self.conv2(x)), kernel_size=2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        return F.log_softmax(self.fc2(x), dim=1)


class GreekNetwork(nn.Module):
    """Transfer learning network: reuses MNIST conv layers (frozen),
    replaces the classification head with a new output layer for 3 classes."""

    def __init__(self):
        super(GreekNetwork, self).__init__()

        # Reused feature extractor from MNIST (weights will be loaded + frozen)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=10, out_channels=20, kernel_size=5)
        self.dropout = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(320, 50)

        # New classification head: 50 → 3 classes (alpha, beta, gamma)
        self.fc2 = nn.Linear(50, 3)

    # Computes a forward pass; returns log_softmax over 3 Greek letter classes
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), kernel_size=2))
        x = F.relu(F.max_pool2d(self.dropout(self.conv2(x)), kernel_size=2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        return F.log_softmax(self.fc2(x), dim=1)


class GreekDataset(Dataset):
    """Custom dataset that loads handwritten Greek letter images from a directory.
    Expects subdirectories named 'alpha', 'beta', 'gamma' containing image files."""

    # Maps folder name to class index
    labels = {"alpha": 0, "beta": 1, "gamma": 2}

    def __init__(self, root_dir):
        self.samples = []  # list of (image_path, label) tuples
        self.transform = transforms.Compose(
            [
                transforms.Grayscale(),
                transforms.Resize((28, 28)),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: 1.0 - x),  # invert to white-on-black
                transforms.Normalize((0.1307,), (0.3081,)),  # match MNIST normalization
            ]
        )

        for letter, idx in GreekDataset.labels.items():
            folder = os.path.join(root_dir, letter)
            for path in glob.glob(os.path.join(folder, "*.png")):
                self.samples.append((path, idx))

        print(f"Loaded {len(self.samples)} Greek letter samples from '{root_dir}'")

    # Returns the number of samples in the dataset
    def __len__(self):
        return len(self.samples)

    # Returns the preprocessed image tensor and label for a given index
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path)
        return self.transform(img), label


# ── Useful Functions ──────────────────────────────────────────────────────────


def load_mnist_weights(greek_model, mnist_path, device):
    """Loads pretrained MNIST weights into the Greek network's shared layers,
    then freezes them so only fc2 is trained."""
    mnist_model = MNISTNetwork()
    mnist_model.load_state_dict(torch.load(mnist_path, map_location=device))

    # Copy weights for all shared layers
    greek_model.conv1.weight.data = mnist_model.conv1.weight.data.clone()
    greek_model.conv1.bias.data = mnist_model.conv1.bias.data.clone()
    greek_model.conv2.weight.data = mnist_model.conv2.weight.data.clone()
    greek_model.conv2.bias.data = mnist_model.conv2.bias.data.clone()
    greek_model.fc1.weight.data = mnist_model.fc1.weight.data.clone()
    greek_model.fc1.bias.data = mnist_model.fc1.bias.data.clone()

    # Freeze all layers except the new classification head (fc2)
    for name, param in greek_model.named_parameters():
        if name.startswith("fc2"):
            param.requires_grad = True
        else:
            param.requires_grad = False

    print(f"MNIST weights loaded from '{mnist_path}' and frozen (fc2 only trainable)\n")
    return greek_model


def train_greek(model, train_loader, optimizer, device, epoch):
    """Runs one training epoch on the Greek letter dataset; returns loss and accuracy."""
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        output = model(images)
        loss = F.nll_loss(output, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = output.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    print(f"Epoch {epoch} — Loss: {avg_loss:.4f}  Acc: {accuracy*100:.2f}%")
    return avg_loss, accuracy


def plot_training_curve(losses, accs, save_path="plot/greek_training.png"):
    """Plots and saves the training loss and accuracy curves."""
    epochs = range(1, len(losses) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.plot(epochs, losses)
    ax1.set_title("Loss")
    ax1.set_xlabel("Epoch")
    ax2.plot(epochs, accs)
    ax2.set_title("Accuracy")
    ax2.set_xlabel("Epoch")

    plt.suptitle("Greek Letter Training", fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Training curve saved to '{save_path}'")
    plt.show()


def plot_greek_predictions(
    model, dataset, device, save_path="plot/greek_predictions.png"
):
    """Plots all Greek letter samples with their predicted and true labels."""
    class_names = ["alpha", "beta", "gamma"]
    n = len(dataset)
    cols = 5
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 3))
    axes = np.array(axes).reshape(-1)

    model.eval()
    for i in range(len(axes)):
        ax = axes[i]
        if i < n:
            img_tensor, true_label = dataset[i]
            with torch.no_grad():
                output = model(img_tensor.unsqueeze(0).to(device))
                pred = output.argmax(dim=1).item()

            # Display the uninverted image for natural appearance
            ax.imshow(img_tensor.squeeze(), cmap="gray")
            color = "green" if pred == true_label else "red"
            ax.set_title(
                f"True: {class_names[true_label]}\nPred: {class_names[pred]}",
                color=color,
                fontsize=8,
            )
        ax.axis("off")

    fig.suptitle("Greek Letter Predictions", fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Predictions plot saved to '{save_path}'")
    plt.show()


# ── Main Function ─────────────────────────────────────────────────────────────


def main(argv):
    # Usage: python3 greek_letters.py [mnist_model_path] [greek_data_dir] [epochs]
    # Example: python3 greek_letters.py model/mnist_cnn.pth greek_data/ 20
    mnist_path = argv[1] if len(argv) > 1 else "model/mnist_cnn.pth"
    data_dir = argv[2] if len(argv) > 2 else "greek_train"
    epochs = int(argv[3]) if len(argv) > 3 else 50

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    # Build Greek network, load pretrained MNIST weights, then move to device
    model = GreekNetwork()
    model = load_mnist_weights(model, mnist_path, device)
    model = model.to(device)

    # Load Greek letter dataset
    # Expects: greek_data/alpha/*.jpg, greek_data/beta/*.jpg, greek_data/gamma/*.jpg
    dataset = GreekDataset(data_dir)
    train_loader = DataLoader(dataset, batch_size=5, shuffle=True)
    print(f"Training on {len(dataset)} samples ({len(dataset)//3} per class)\n")

    # Only train the new classification head (fc2)
    # Use a small learning rate — the frozen layers are already well-tuned
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=0.0001
    )

    # Training loop
    losses, accs = [], []
    for epoch in range(1, epochs + 1):
        loss, acc = train_greek(model, train_loader, optimizer, device, epoch)
        losses.append(loss)
        accs.append(acc)

    # Plot training curves
    plot_training_curve(losses, accs)

    # Show predictions on all training samples
    plot_greek_predictions(model, dataset, device)

    # Save the trained Greek model
    save_path = "model/greek_cnn.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to '{save_path}'")

    return


if __name__ == "__main__":
    main(sys.argv)
