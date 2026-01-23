"""
===========================================================
Recall control for semantic segmentation with MAPIE
===========================================================

This example illustrates how to control the recall of a
semantic segmentation model using MAPIE.

We use :class:`~mapie.risk_control.SemanticSegmentationController`
to calibrate a decision threshold that statistically guarantees
a target recall level on unseen data.

The dataset, model and utility functions are loaded from
Hugging Face for simplicity and reproducibility.
"""

import importlib.util
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from huggingface_hub import hf_hub_download, snapshot_download

from mapie.risk_control import SemanticSegmentationController

warnings.filterwarnings("ignore")

###############################################################################
# To keep this example self-contained, we load the dataset utilities
# and the segmentation LightningModule definition directly from a
# repository hosted on Hugging Face.
#

module_path = hf_hub_download(
    repo_id="mapie-library/rooftop_segmentation",
    filename="model_and_lightning_module.py",
    repo_type="dataset",
)
spec = importlib.util.spec_from_file_location("hf_module", module_path)
hf_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hf_module)

SegmentationLightningModule = hf_module.SegmentationLightningModule
RoofSegmentationDataset = hf_module.RoofSegmentationDataset
get_validation_transforms = hf_module.get_validation_transforms


###############################################################################
# Load a pretrained segmentation model checkpoint from Hugging Face.
#

model_ckpt = hf_hub_download(
    repo_id="mapie-library/rooftop_segmentation",
    filename="best_model-v1.ckpt",
    repo_type="dataset",
)

data_root = Path(
    snapshot_download(
        repo_id="mapie-library/rooftop_segmentation",
        repo_type="dataset",
        allow_patterns=["calib/**", "test/**"],
    )
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model = SegmentationLightningModule.load_from_checkpoint(model_ckpt)
model.to(DEVICE)
model.eval()
print("Model loaded successfully!")

###############################################################################
# Next, two datasets are loaded from Hugging Face: a calibration set used to estimate
# risks and select an appropriate decision threshold, and a test set reserved for
# evaluating controlled predictions on unseen data.
#

CALIB_IMAGES_DIR = data_root / "calib" / "images"
CALIB_MASKS_DIR = data_root / "calib" / "masks"
TEST_IMAGES_DIR = data_root / "test" / "images"
TEST_MASKS_DIR = data_root / "test" / "masks"

calib_dataset = RoofSegmentationDataset(
    images_dir=CALIB_IMAGES_DIR,
    masks_dir=CALIB_MASKS_DIR,
    transform=get_validation_transforms(),
)
calib_loader = torch.utils.data.DataLoader(calib_dataset, batch_size=8)

test_dataset = RoofSegmentationDataset(
    images_dir=TEST_IMAGES_DIR,
    masks_dir=TEST_MASKS_DIR,
    transform=get_validation_transforms(),
)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=8)

print(f"Calibration set size: {len(calib_dataset)}")
print(f"Test set size: {len(test_dataset)}")

###############################################################################
# A :class:`~mapie.risk_control.SemanticSegmentationController` is instantiated
# to control the recall risk (1 - recall) and automatically select a threshold
# that meets the target recall level with high confidence.
#

TARGET_RECALL = 0.9
recall_controller = SemanticSegmentationController(
    predict_function=model,
    risk="recall",
    target_level=TARGET_RECALL,
    confidence_level=None,
)

print(f"Target recall level: {TARGET_RECALL}")

###############################################################################
# During calibration, the controller evaluates the recall risk over a range
# of thresholds on the calibration dataset in order to identify an optimal
# decision threshold.
#

for i, sample in enumerate(calib_loader):
    image, mask = sample["image"], sample["mask"]
    image = image.to(DEVICE)
    mask = mask.cpu().numpy()

    # Filter images that contain masks
    has_mask = mask.sum(axis=(1, 2)) > 0
    image = image[has_mask]
    mask = mask[has_mask]

    if len(image) > 0:
        with torch.no_grad():
            recall_controller.compute_risks(image, mask)

# Compute the best threshold
recall_controller.compute_best_predict_param()
print("Controller calibrated successfully!")
print(f"Optimal threshold found: {recall_controller.best_predict_param[0]:.4f}")

###############################################################################
# Controlled predictions are visually inspected on a few test images to
# illustrate the effect of MAPIE thresholding compared to raw model outputs.
#


def denormalize_image(tensor_image: torch.Tensor) -> np.ndarray:
    """Denormalize image tensor for visualization."""
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    image = tensor_image.cpu().numpy().transpose(1, 2, 0)
    image = std * image + mean
    image = np.clip(image, 0, 1)

    return image


# Select random test images
NUM_EXAMPLES = 4
np.random.seed(42)

# Get indices of images with masks
indices_with_masks = []
for idx in range(len(test_dataset)):
    sample = test_dataset[idx]
    if sample["mask"].sum() > 0:
        indices_with_masks.append(idx)

random_indices = np.random.choice(indices_with_masks, NUM_EXAMPLES, replace=False)

fig, axes = plt.subplots(NUM_EXAMPLES, 4, figsize=(16, 4 * NUM_EXAMPLES))

for row, idx in enumerate(random_indices):
    sample = test_dataset[idx]
    image = sample["image"].unsqueeze(0).to(DEVICE)
    mask = sample["mask"].cpu().numpy()

    with torch.no_grad():
        # Get unthresholded prediction
        logits = model(image)
        prob = torch.sigmoid(logits).cpu().numpy()[0, 0]

        # Get MAPIE prediction
        mapie_pred = recall_controller.predict(image)[0]

    # Denormalize image
    img_display = denormalize_image(sample["image"])

    # Plot
    axes[row, 0].imshow(img_display)
    axes[row, 0].set_title("Original Image")
    axes[row, 0].axis("off")

    axes[row, 1].imshow(mask, cmap="gray")
    axes[row, 1].set_title("Ground Truth")
    axes[row, 1].axis("off")

    axes[row, 2].imshow(prob, cmap="viridis", vmin=0, vmax=1)
    axes[row, 2].set_title("Prediction Probability")
    axes[row, 2].axis("off")

    axes[row, 3].imshow(mapie_pred[0], cmap="gray")
    axes[row, 3].set_title(
        f"MAPIE Prediction\n(threshold={recall_controller.best_predict_param[0]:.3f})"
    )
    axes[row, 3].axis("off")

plt.tight_layout()
plt.show()

###############################################################################
# The controller is finally evaluated on the test set by computing the achieved
# recall on each image to verify that the target recall level is satisfied
# on unseen data.
#

recalls_list = []

for i, sample in enumerate(test_loader):
    image, mask = sample["image"], sample["mask"]
    image = image.to(DEVICE)
    mask = mask.cpu().numpy()

    # Filter images with masks
    has_mask = mask.sum(axis=(1, 2)) > 0
    image = image[has_mask]
    mask = mask[has_mask]

    if len(image) > 0:
        with torch.no_grad():
            pred = recall_controller.predict(image)

            # Compute recall for each image
            for j in range(len(image)):
                tp = (mask[j] * pred[j]).sum()
                fn = (mask[j] * (1 - pred[j])).sum()
                recall = tp / (tp + fn + 1e-8)
                recalls_list.append(recall)

recalls_array = np.array(recalls_list)

###############################################################################
# Finally, the distribution of recall values over the test set is plotted
# to summarize the controlled performance.
#

fig, ax = plt.subplots(figsize=(10, 6))

ax.hist(recalls_array, bins=30, alpha=0.7, color="steelblue", edgecolor="black")
ax.axvline(
    TARGET_RECALL,
    color="red",
    linestyle="--",
    linewidth=2,
    label=f"Target Recall ({TARGET_RECALL})",
)
ax.axvline(
    recalls_array.mean(),
    color="orange",
    linestyle="--",
    linewidth=2,
    label=f"Mean Recall ({recalls_array.mean():.3f})",
)
ax.set_xlabel("Recall", fontsize=12)
ax.set_ylabel("Frequency", fontsize=12)
ax.set_title("Distribution of Recall on Test Set", fontsize=14, fontweight="bold")
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

###############################################################################
# The histogram shows that most test images achieve or exceed the
# target recall level, illustrating the effectiveness of MAPIE’s
# risk control for semantic segmentation tasks.
#

###############################################################################
# Bootstrap the mean recall over different samplings of the test set
# (resampling images with replacement).
#

N_BOOTSTRAP = 2000
BOOTSTRAP_SEED = 123
rng = np.random.default_rng(BOOTSTRAP_SEED)

bootstrap_means = np.empty(N_BOOTSTRAP, dtype=float)
n = recalls_array.size
for b in range(N_BOOTSTRAP):
    bootstrap_sample = rng.choice(recalls_array, size=n, replace=True)
    bootstrap_means[b] = bootstrap_sample.mean()

fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(
    bootstrap_means,
    bins=40,
    alpha=0.7,
    color="slateblue",
    edgecolor="black",
)
ax.axvline(
    bootstrap_means.mean(),
    color="green",
    linestyle="--",
    linewidth=2,
    label=f"Bootstrap mean ({bootstrap_means.mean():.3f})",
)
ax.axvline(
    TARGET_RECALL,
    color="orange",
    linestyle="--",
    linewidth=2,
    label=f"Target recall ({TARGET_RECALL:.2f})",
)
ax.set_xlabel("Bootstrap mean recall", fontsize=12)
ax.set_ylabel("Frequency", fontsize=12)
ax.set_title(
    "Bootstrap distribution of mean recall (test set resampling)",
    fontsize=14,
    fontweight="bold",
)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
