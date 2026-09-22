import cv2
import numpy as np
import torch

from PIL import Image
from transformers import (
    AutoImageProcessor,
    SegformerForSemanticSegmentation
)

# --------------------------------
# FILES / SETTINGS
# --------------------------------

INPUT_FILE = "nearmap_test.jpg"
PROBABILITY_FILE = "water_probability.jpg"
SEED_FILE = "water_seed_mask.jpg"
MASK_FILE = "water_mask_ai.jpg"
OUTPUT_FILE = "shoreline_ai.jpg"

MODEL_NAME = "nvidia/segformer-b0-finetuned-ade-512-512"

# Two-threshold strategy:
# HIGH threshold = very confident water
# LOW threshold  = possible water
HIGH_THRESHOLD = 0.28
LOW_THRESHOLD = 0.14

# Remove tiny blobs
MIN_COMPONENT_AREA = 2500

WATER_LABELS = {
    "water",
    "sea",
    "river",
    "lake",
    "swimming pool",
    "waterfall"
}


# --------------------------------
# HELPERS
# --------------------------------

def touches_border(component_mask):
    """Return True if white pixels touch any image border."""
    if np.any(component_mask[0, :] > 0):
        return True
    if np.any(component_mask[-1, :] > 0):
        return True
    if np.any(component_mask[:, 0] > 0):
        return True
    if np.any(component_mask[:, -1] > 0):
        return True
    return False


# --------------------------------
# LOAD IMAGE
# --------------------------------

print()
print("AI SHORELINE DETECTOR V3")
print("------------------------")

image_pil = Image.open(INPUT_FILE).convert("RGB")
width, height = image_pil.size

print(f"Image: {width} x {height}")


# --------------------------------
# LOAD MODEL
# --------------------------------

print()
print("Loading SegFormer...")

processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
model = SegformerForSemanticSegmentation.from_pretrained(MODEL_NAME)
model.eval()

print("Model loaded.")


# --------------------------------
# RUN MODEL
# --------------------------------

print("Running segmentation...")

inputs = processor(images=image_pil, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

logits = torch.nn.functional.interpolate(
    outputs.logits,
    size=(height, width),
    mode="bilinear",
    align_corners=False
)

probabilities = torch.softmax(logits, dim=1)[0]


# --------------------------------
# FIND WATER CLASS IDS
# --------------------------------

water_ids = []

print()
print("Water classes found:")

for class_id, label in model.config.id2label.items():
    label_clean = label.strip().lower()
    if label_clean in WATER_LABELS:
        water_ids.append(int(class_id))
        print(f"  {class_id}: {label}")

if not water_ids:
    raise RuntimeError("No water classes found.")


# --------------------------------
# COMBINE WATER PROBABILITIES
# --------------------------------

water_probability = torch.zeros((height, width))

for water_id in water_ids:
    water_probability += probabilities[water_id]

water_probability = water_probability.cpu().numpy()


# --------------------------------
# SAVE PROBABILITY IMAGE
# --------------------------------

probability_image = (water_probability * 255).clip(0, 255).astype(np.uint8)
cv2.imwrite(PROBABILITY_FILE, probability_image)


# --------------------------------
# HIGH / LOW WATER MASKS
# --------------------------------

high_mask = (water_probability >= HIGH_THRESHOLD).astype(np.uint8) * 255
low_mask = (water_probability >= LOW_THRESHOLD).astype(np.uint8) * 255

# Clean each mask a bit
small_kernel = np.ones((5, 5), np.uint8)

high_mask = cv2.morphologyEx(high_mask, cv2.MORPH_OPEN, small_kernel)
high_mask = cv2.morphologyEx(high_mask, cv2.MORPH_CLOSE, small_kernel)

low_mask = cv2.morphologyEx(low_mask, cv2.MORPH_OPEN, small_kernel)
low_mask = cv2.morphologyEx(low_mask, cv2.MORPH_CLOSE, small_kernel)


# --------------------------------
# FIND HIGH-CONFIDENCE SEED WATER
# Only keep seed components that touch the border
# since the main waterway should run into/out of the image
# --------------------------------

num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
    high_mask,
    connectivity=8
)

seed_mask = np.zeros_like(high_mask)

print()
print("High-confidence seed components:")

for label_id in range(1, num_labels):
    area = stats[label_id, cv2.CC_STAT_AREA]

    component = np.zeros_like(high_mask)
    component[labels == label_id] = 255

    border = touches_border(component)

    print(
        f"Seed region {label_id}: "
        f"{area} pixels | border={border}"
    )

    if area >= MIN_COMPONENT_AREA and border:
        seed_mask[labels == label_id] = 255

cv2.imwrite(SEED_FILE, seed_mask)


# --------------------------------
# GROW SEEDS THROUGH LOW-CONFIDENCE WATER
# Keep only low-mask components that overlap seed_mask
# --------------------------------

num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
    low_mask,
    connectivity=8
)

grown_mask = np.zeros_like(low_mask)

print()
print("Low-confidence components:")

for label_id in range(1, num_labels):
    area = stats[label_id, cv2.CC_STAT_AREA]

    component_region = (labels == label_id)
    overlap = np.any(seed_mask[component_region] > 0)

    print(
        f"Low region {label_id}: "
        f"{area} pixels | overlaps_seed={overlap}"
    )

    if area >= MIN_COMPONENT_AREA and overlap:
        grown_mask[component_region] = 255


# --------------------------------
# FINAL CLEANUP
# --------------------------------

merge_kernel = np.ones((13, 13), np.uint8)
grown_mask = cv2.morphologyEx(grown_mask, cv2.MORPH_CLOSE, merge_kernel)

# Fill small gaps
final_kernel = np.ones((9, 9), np.uint8)
grown_mask = cv2.morphologyEx(grown_mask, cv2.MORPH_CLOSE, final_kernel)

# Optional slight smoothing
grown_mask = cv2.medianBlur(grown_mask, 5)


# --------------------------------
# FIND SHORELINE
# --------------------------------

contours, _ = cv2.findContours(
    grown_mask,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

original = cv2.imread(INPUT_FILE)
result = original.copy()

cv2.drawContours(
    result,
    contours,
    -1,
    (0, 0, 255),
    3
)


# --------------------------------
# SAVE RESULTS
# --------------------------------

cv2.imwrite(MASK_FILE, grown_mask)
cv2.imwrite(OUTPUT_FILE, result)

print()
print("------------------------")
print("SUCCESS!")
print(f"Water probability: {PROBABILITY_FILE}")
print(f"Seed mask:          {SEED_FILE}")
print(f"Final water mask:   {MASK_FILE}")
print(f"Shoreline:          {OUTPUT_FILE}")
print("------------------------")