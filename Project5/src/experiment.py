# John Doe — Experiments sweeping patch size/stride, depth vs width, and
#             CLS token vs mean pooling on the Vision Transformer for MNIST

# import statements
import sys
import time
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


# ── Class Definitions ────────────────────────────────────────────────────────


class PatchEmbedding(nn.Module):
    """Converts an image into overlapping or non-overlapping patch token embeddings."""

    def __init__(self, image_size, patch_size, stride, in_channels, embed_dim):
        super().__init__()
        self.patch_size = patch_size
        self.stride = stride
        self.unfold = nn.Unfold(kernel_size=patch_size, stride=stride)
        self.patch_dim = in_channels * patch_size * patch_size
        self.proj = nn.Linear(self.patch_dim, embed_dim)
        positions = ((image_size - patch_size) // stride) + 1
        self.num_patches = positions * positions

    # Extracts patches and projects each to embed_dim
    def forward(self, x):
        x = self.unfold(x)  # (B, patch_dim, N)
        x = x.transpose(1, 2)  # (B, N, patch_dim)
        x = self.proj(x)  # (B, N, embed_dim)
        return x


class ViTNet(nn.Module):
    """Configurable Vision Transformer supporting variable depth, width,
    patch size, stride, and aggregation strategy (CLS token or mean pooling)."""

    def __init__(
        self,
        patch_size=4,
        stride=2,
        embed_dim=48,
        depth=4,
        num_heads=8,
        mlp_dim=128,
        dropout=0.1,
        use_cls_token=False,
        num_classes=10,
    ):
        super().__init__()

        self.use_cls_token = use_cls_token
        self.patch_embed = PatchEmbedding(28, patch_size, stride, 1, embed_dim)
        num_tokens = self.patch_embed.num_patches + (1 if use_cls_token else 0)

        if use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        else:
            self.cls_token = None

        self.pos_embed = nn.Parameter(torch.randn(1, num_tokens, embed_dim) * 0.02)
        self.pos_drop = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=mlp_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim), nn.GELU(), nn.Linear(mlp_dim, num_classes)
        )

        self._init_weights()

    # Initializes positional and CLS token weights for stable training
    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        if self.cls_token is not None:
            nn.init.trunc_normal_(self.cls_token, std=0.02)

    # Computes a forward pass through patch embedding, transformer, and classifier
    def forward(self, x):
        B = x.size(0)
        x = self.patch_embed(x)
        if self.use_cls_token:
            cls = self.cls_token.expand(B, -1, -1)
            x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        x = self.encoder(x)
        x = self.norm(x)
        x = x[:, 0] if self.use_cls_token else x.mean(dim=1)
        return F.log_softmax(self.classifier(x), dim=1)


# ── Useful Functions ──────────────────────────────────────────────────────────


def load_data(batch_size=64):
    """Loads and returns MNIST train and test DataLoaders."""
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    train_set = datasets.MNIST("./data", train=True, download=True, transform=transform)
    test_set = datasets.MNIST("./data", train=False, download=True, transform=transform)
    return (
        DataLoader(train_set, batch_size=batch_size, shuffle=True),
        DataLoader(test_set, batch_size=batch_size, shuffle=False),
    )


def run_experiment(config, train_loader, test_loader, device, epochs=10):
    """Trains and evaluates a ViTNet with the given config dict.
    Returns lists of per-epoch train accuracy, test accuracy, and epoch times."""
    model = ViTNet(
        **{
            k: config[k]
            for k in [
                "patch_size",
                "stride",
                "embed_dim",
                "depth",
                "num_heads",
                "mlp_dim",
                "dropout",
                "use_cls_token",
            ]
        }
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    train_accs, test_accs, epoch_times = [], [], []

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        # training pass
        model.train()
        correct, total = 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            out = model(images)
            loss = F.nll_loss(out, labels)
            loss.backward()
            optimizer.step()
            correct += (out.argmax(1) == labels).sum().item()
            total += labels.size(0)
        train_accs.append(correct / total * 100)

        # evaluation pass
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                out = model(images)
                correct += (out.argmax(1) == labels).sum().item()
                total += labels.size(0)
        test_accs.append(correct / total * 100)
        scheduler.step()

        epoch_times.append(time.time() - t0)
        print(
            f"  [{config['label']}] Epoch {epoch}/{epochs} | "
            f"Train: {train_accs[-1]:.2f}%  Test: {test_accs[-1]:.2f}%  "
            f"Time: {epoch_times[-1]:.1f}s"
        )

    return train_accs, test_accs, epoch_times


def save_results_csv(all_results, path="experiment_results.csv"):
    """Saves all experiment results to a CSV file."""
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "experiment",
                "label",
                "best_test_acc",
                "avg_epoch_time_s",
                "final_train_acc",
            ]
        )
        for exp_name, configs in all_results.items():
            for cfg, res in configs:
                writer.writerow(
                    [
                        exp_name,
                        cfg["label"],
                        f"{max(res['test_accs']):.2f}",
                        f"{sum(res['epoch_times'])/len(res['epoch_times']):.2f}",
                        f"{res['train_accs'][-1]:.2f}",
                    ]
                )
    print(f"Results saved to '{path}'")


def plot_experiment(configs_results, title, save_path, epochs):
    """Plots test accuracy and avg epoch time side by side for one experiment dimension."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))
    fig.suptitle(title, fontsize=13)
    ep = range(1, epochs + 1)

    for cfg, res in configs_results:
        ax1.plot(ep, res["test_accs"], label=cfg["label"])
        ax2.bar(
            cfg["label"], sum(res["epoch_times"]) / len(res["epoch_times"]), width=0.5
        )

    ax1.set_title("Test Accuracy (%)")
    ax1.set_xlabel("Epoch")
    ax1.legend()
    ax2.set_title("Avg Epoch Time (s)")
    ax2.set_xlabel("Config")
    ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Plot saved to '{save_path}'")
    plt.show()


def plot_summary(all_results, save_path="plot/experiment_summary.png"):
    """Plots a bar chart comparing best test accuracy across all experiment configs."""
    labels, accs, times = [], [], []
    for configs_results in all_results.values():
        for cfg, res in configs_results:
            labels.append(cfg["label"])
            accs.append(max(res["test_accs"]))
            times.append(sum(res["epoch_times"]) / len(res["epoch_times"]))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Summary — All Experiments", fontsize=13)

    x = range(len(labels))
    ax1.bar(x, accs, width=0.6)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax1.set_title("Best Test Accuracy (%)")
    ax1.set_ylabel("%")

    ax2.bar(x, times, width=0.6)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax2.set_title("Avg Epoch Time (s)")
    ax2.set_ylabel("seconds")

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Summary plot saved to '{save_path}'")
    plt.show()


# ── Main Function ─────────────────────────────────────────────────────────────


def main(argv):
    # Usage: python3 experiment.py [epochs]
    epochs = int(argv[1]) if len(argv) > 1 else 5

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  Epochs per config: {epochs}\n")

    train_loader, test_loader = load_data()

    # base config shared across all experiments
    base = dict(
        patch_size=4,
        stride=2,
        embed_dim=48,
        depth=4,
        num_heads=8,
        mlp_dim=128,
        dropout=0.1,
        use_cls_token=False,
    )

    # ── Experiment 1: Patch size & stride ─────────────────────────────────────
    # Note: patch=2 stride=1 produces 729 tokens and is extremely slow — excluded
    exp1_configs = [
        {
            **base,
            "patch_size": 7,
            "stride": 7,
            "label": "patch=7 stride=7 (no overlap)",
        },
        {**base, "patch_size": 4, "stride": 2, "label": "patch=4 stride=2 (overlap)"},
        {**base, "patch_size": 3, "stride": 2, "label": "patch=3 stride=2 (fine)"},
    ]

    # ── Experiment 2: Depth vs width ──────────────────────────────────────────
    exp2_configs = [
        {
            **base,
            "depth": 2,
            "embed_dim": 96,
            "mlp_dim": 256,
            "num_heads": 8,
            "label": "shallow-wide  d=2 dim=96",
        },
        {
            **base,
            "depth": 4,
            "embed_dim": 48,
            "mlp_dim": 128,
            "num_heads": 8,
            "label": "balanced     d=4 dim=48",
        },
        {
            **base,
            "depth": 8,
            "embed_dim": 32,
            "mlp_dim": 64,
            "num_heads": 8,
            "label": "deep-narrow  d=8 dim=32",
        },
    ]

    # ── Experiment 3: CLS token vs mean pooling ───────────────────────────────
    exp3_configs = [
        {**base, "use_cls_token": False, "label": "mean pooling"},
        {**base, "use_cls_token": True, "label": "CLS token"},
    ]

    experiments = {
        "Patch Size & Stride": exp1_configs,
        "Depth vs Width": exp2_configs,
        "CLS Token vs Mean Pooling": exp3_configs,
    }

    all_results = {}

    for exp_name, configs in experiments.items():
        print("=" * 60)
        print(f"Experiment: {exp_name}")
        print("=" * 60)

        configs_results = []
        for cfg in configs:
            print(f"\n--- Config: {cfg['label']} ---")
            train_accs, test_accs, epoch_times = run_experiment(
                cfg, train_loader, test_loader, device, epochs
            )
            configs_results.append(
                (
                    cfg,
                    {
                        "train_accs": train_accs,
                        "test_accs": test_accs,
                        "epoch_times": epoch_times,
                    },
                )
            )

        all_results[exp_name] = configs_results

        # plot results for this experiment dimension
        safe_name = exp_name.lower().replace(" ", "_").replace("&", "and")
        plot_experiment(configs_results, exp_name, f"plot/exp_{safe_name}.png", epochs)

    # summary plot and CSV across all experiments
    plot_summary(all_results)
    save_results_csv(all_results)

    return


if __name__ == "__main__":
    main(sys.argv)
