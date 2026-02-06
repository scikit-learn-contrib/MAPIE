"""
===========================================
Precision control for semantic segmentation
===========================================

This example illustrates how to control the precision of a
semantic segmentation model using MAPIE.

We use :class:`~mapie.risk_control.SemanticSegmentationController`
to calibrate a decision threshold that statistically guarantees
a target precision level on unseen data.

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
    transform=get_validation_transforms(
        image_size=(256, 256)
    ),  # reshape images to reduce memory usage
)
calib_loader = torch.utils.data.DataLoader(calib_dataset, batch_size=16)

test_dataset = RoofSegmentationDataset(
    images_dir=TEST_IMAGES_DIR,
    masks_dir=TEST_MASKS_DIR,
    transform=get_validation_transforms(
        image_size=(256, 256)
    ),  # reshape images to reduce memory usage
)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16)

print(f"Calibration set size: {len(calib_dataset)}")
print(f"Test set size: {len(test_dataset)}")

###############################################################################
# A :class:`~mapie.risk_control.SemanticSegmentationController` is instantiated
# to control the precision risk (1 - precision) and automatically select a threshold
# that meets the target precision level with high confidence.
#

TARGET_PRECISION = 0.7
CONFIDENCE_LEVEL = 0.9
precision_controller = SemanticSegmentationController(
    predict_function=model,
    risk="precision",
    target_level=TARGET_PRECISION,
    confidence_level=CONFIDENCE_LEVEL,
)

print(f"Target precision level: {TARGET_PRECISION}")

###############################################################################
# During calibration, the controller evaluates the precision risk over a range
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
            precision_controller.compute_risks(image, mask)

# Compute the best threshold
precision_controller.compute_best_predict_param()
print("Controller calibrated successfully!")
print(f"Optimal threshold found: {precision_controller.best_predict_param[0]:.4f}")

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

fig, axes = plt.subplots(2, NUM_EXAMPLES, figsize=(4 * NUM_EXAMPLES, 10))

for col, idx in enumerate(random_indices):
    sample = test_dataset[idx]
    image = sample["image"].unsqueeze(0).to(DEVICE)
    mask = sample["mask"].cpu().numpy()

    with torch.no_grad():
        # Get MAPIE prediction
        mapie_pred = precision_controller.predict(image)[0]

    # Denormalize image
    img_display = denormalize_image(sample["image"])

    # Plot original image (top row)
    axes[0, col].imshow(img_display)
    axes[0, col].set_title("Original Image")
    axes[0, col].axis("off")

    # Plot MAPIE prediction with correct pixels in white and false positives in red (bottom row)
    pred_visualization = np.zeros((*mapie_pred[0].shape, 3))
    true_positives = mask * mapie_pred[0]
    pred_visualization[true_positives > 0] = [1, 1, 1]
    false_positives = (1 - mask) * mapie_pred[0]
    pred_visualization[false_positives > 0] = [1, 0, 0]

    axes[1, col].imshow(pred_visualization)
    axes[1, col].set_title(
        f"MAPIE Prediction (threshold={precision_controller.best_predict_param[0]:.2f})\n"
        "White: Correct | Red: False Positives"
    )
    axes[1, col].axis("off")

plt.tight_layout()
plt.show()


###############################################################################
# The controller is finally evaluated on the test set by computing the achieved
# precision on each image to verify that the target precision level is satisfied
# on unseen data.
#

precisions_list = []

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
            pred = precision_controller.predict(image)

            # Compute precision for each image
            for j in range(len(image)):
                tp = (mask[j] * pred[j]).sum()
                fp = ((1 - mask[j]) * pred[j]).sum()
                precision = tp / (tp + fp + 1e-8)
                precisions_list.append(precision)

precisions_array = np.array(precisions_list)

###############################################################################
# Finally, the distribution of precision values over the test set is plotted
# to summarize the controlled performance.
#

fig, ax = plt.subplots(figsize=(10, 6))

ax.hist(precisions_array, bins=30, alpha=0.7, color="steelblue", edgecolor="black")
ax.axvline(
    TARGET_PRECISION,
    color="red",
    linestyle="--",
    linewidth=2,
    label=f"Target Precision ({TARGET_PRECISION})",
)
ax.axvline(
    precisions_array.mean(),
    color="orange",
    linestyle="--",
    linewidth=2,
    label=f"Mean Precision ({precisions_array.mean():.3f})",
)
ax.set_xlabel("Precision", fontsize=12)
ax.set_ylabel("Frequency", fontsize=12)
ax.set_title("Distribution of Precision on Test Set", fontsize=14, fontweight="bold")
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

###############################################################################
# The histogram shows that most test images achieve or exceed the
# target precision level, illustrating the effectiveness of MAPIE’s
# risk control for semantic segmentation tasks.
#

###############################################################################
# Bootstrap the mean precision over different samplings of the test set
# (resampling images with replacement).
#

N_BOOTSTRAP = 2000
BOOTSTRAP_SEED = 123
rng = np.random.default_rng(BOOTSTRAP_SEED)

bootstrap_means = np.empty(N_BOOTSTRAP, dtype=float)
n = precisions_array.size
for b in range(N_BOOTSTRAP):
    bootstrap_sample = rng.choice(precisions_array, size=n, replace=True)
    bootstrap_means[b] = bootstrap_sample.mean()

delta = round(1 - CONFIDENCE_LEVEL, 2)
quantile_confidence = np.quantile(bootstrap_means, delta)
print(f"Bootstrap {delta}-th quantile: {quantile_confidence:.4f}")

fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(
    bootstrap_means,
    bins=40,
    alpha=0.7,
    color="slateblue",
    edgecolor="black",
)
ax.axvline(
    TARGET_PRECISION,
    color="orange",
    linestyle="--",
    linewidth=2,
    label=f"Target precision ({TARGET_PRECISION:.2f})",
)
ax.axvline(
    quantile_confidence,
    color="green",
    linestyle="--",
    linewidth=2,
    label=f"Bootstrap {delta}-th quantile ({quantile_confidence:.3f})",
)
ax.set_xlabel("Bootstrap mean precision", fontsize=12)
ax.set_ylabel("Frequency", fontsize=12)
ax.set_title(
    "Bootstrap distribution of mean precision (test set resampling)",
    fontsize=14,
    fontweight="bold",
)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
