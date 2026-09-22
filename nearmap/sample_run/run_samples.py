import os
import math
import json
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
import requests
import torch

from PIL import Image
from dotenv import load_dotenv
from transformers import (
    AutoImageProcessor,
    SegformerForSemanticSegmentation
)


# ============================================================
# SETTINGS
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent

# Try .env in sample_run first, then parent nearmap folder
load_dotenv(SCRIPT_DIR / ".env")
load_dotenv(PROJECT_DIR / ".env")

API_KEY = os.getenv("NEARMAP_API_KEY")

if not API_KEY:
    raise ValueError(
        "NEARMAP_API_KEY not found. "
        "Expected .env in sample_run or parent nearmap folder."
    )

NUM_SAMPLES = 5

ZOOM = 19
TILE_SIZE = 256

HIGH_THRESHOLD = 0.28
LOW_THRESHOLD = 0.14
MIN_COMPONENT_AREA = 2500

MODEL_NAME = (
    "nvidia/"
    "segformer-b0-finetuned-ade-512-512"
)

WATER_LABELS = {
    "water",
    "sea",
    "river",
    "lake",
    "swimming pool",
    "waterfall"
}

OUTPUT_ROOT = SCRIPT_DIR / "samples"

VIMS_URL = (
    "https://mobjack.vims.edu/ArcGIS/rest/services/"
    "VIMS_SSP/All_WebShoreline1937_1949_2009_2017/"
    "FeatureServer/3/query"
)


# ============================================================
# VIMS SHORELINE
# ============================================================

def get_shoreline_locations():
    print()
    print("Getting shoreline locations from VIMS...")

    params = {
        "where": "1=1",
        "outFields": "*",
        "returnGeometry": "true",
        "outSR": "4326",
        "resultRecordCount": 30,
        "f": "geojson"
    }

    response = requests.get(
        VIMS_URL,
        params=params,
        timeout=30
    )

    response.raise_for_status()

    data = response.json()

    features = data.get("features", [])

    print(
        f"Downloaded {len(features)} "
        "shoreline features."
    )

    candidates = []

    for feature in features:

        geometry = feature.get("geometry")

        if not geometry:
            continue

        geometry_type = geometry.get("type")
        coordinates = geometry.get(
            "coordinates",
            []
        )

        points = []

        if geometry_type == "LineString":

            points = coordinates

        elif geometry_type == "MultiLineString":

            for line in coordinates:
                points.extend(line)

        if not points:
            continue

        # Choose middle coordinate from this
        # shoreline feature
        midpoint = points[len(points) // 2]

        lon = midpoint[0]
        lat = midpoint[1]

        # Basic Virginia-area sanity check
        if (
            36.0 <= lat <= 39.0
            and -78.0 <= lon <= -75.0
        ):
            candidates.append(
                (lat, lon)
            )

    if len(candidates) < NUM_SAMPLES:
        raise RuntimeError(
            "Not enough shoreline locations found."
        )

    # Spread selections across the candidate list
    indexes = np.linspace(
        0,
        len(candidates) - 1,
        NUM_SAMPLES,
        dtype=int
    )

    selected = [
        candidates[i]
        for i in indexes
    ]

    print(
        f"Selected {len(selected)} "
        "shoreline locations."
    )

    return selected


# ============================================================
# TILE MATH
# ============================================================

def lat_lon_to_tile(lat, lon, zoom):
    lat_rad = math.radians(lat)

    n = 2 ** zoom

    x = int(
        (lon + 180.0)
        / 360.0
        * n
    )

    y = int(
        (
            1.0
            - math.asinh(
                math.tan(lat_rad)
            )
            / math.pi
        )
        / 2.0
        * n
    )

    return x, y


# ============================================================
# DOWNLOAD NEARMAP 3x3
# ============================================================

def download_nearmap_image(
    lat,
    lon,
    output_file
):

    center_x, center_y = (
        lat_lon_to_tile(
            lat,
            lon,
            ZOOM
        )
    )

    final_image = Image.new(
        "RGB",
        (
            TILE_SIZE * 3,
            TILE_SIZE * 3
        )
    )

    for row, dy in enumerate(
        [-1, 0, 1]
    ):

        for col, dx in enumerate(
            [-1, 0, 1]
        ):

            x = center_x + dx
            y = center_y + dy

            url = (
                "https://us0.nearmap.com/maps/"
                f"z={ZOOM}"
                f"&x={x}"
                f"&y={y}"
                "&version=2"
                "&nml=Vert"
                "&client=vims_nearmap"
                "&httpauth=false"
                f"&apikey={API_KEY}"
            )

            response = requests.get(
                url,
                timeout=30
            )

            content_type = (
                response.headers.get(
                    "Content-Type",
                    ""
                )
            )

            if (
                response.status_code != 200
                or not
                content_type.startswith(
                    "image/"
                )
            ):
                raise RuntimeError(
                    "Nearmap tile failed: "
                    f"status="
                    f"{response.status_code}"
                )

            tile = Image.open(
                BytesIO(
                    response.content
                )
            ).convert("RGB")

            final_image.paste(
                tile,
                (
                    col * TILE_SIZE,
                    row * TILE_SIZE
                )
            )

    final_image.save(
        output_file,
        "JPEG",
        quality=95
    )


# ============================================================
# MASK HELPERS
# ============================================================

def touches_border(mask):

    return (
        np.any(mask[0, :] > 0)
        or np.any(mask[-1, :] > 0)
        or np.any(mask[:, 0] > 0)
        or np.any(mask[:, -1] > 0)
    )


# ============================================================
# SEGMENT SHORELINE
# ============================================================

def segment_image(
    image_path,
    sample_folder,
    processor,
    model,
    water_ids
):

    image_pil = Image.open(
        image_path
    ).convert("RGB")

    width, height = (
        image_pil.size
    )

    inputs = processor(
        images=image_pil,
        return_tensors="pt"
    )

    with torch.no_grad():
        outputs = model(
            **inputs
        )

    logits = (
        torch.nn.functional.interpolate(
            outputs.logits,
            size=(height, width),
            mode="bilinear",
            align_corners=False
        )
    )

    probabilities = torch.softmax(
        logits,
        dim=1
    )[0]

    water_probability = torch.zeros(
        (height, width)
    )

    for water_id in water_ids:

        water_probability += (
            probabilities[
                water_id
            ]
        )

    water_probability = (
        water_probability
        .cpu()
        .numpy()
    )

    # --------------------------------
    # SAVE PROBABILITY IMAGE
    # --------------------------------

    probability_image = (
        water_probability
        * 255
    ).clip(
        0,
        255
    ).astype(
        np.uint8
    )

    cv2.imwrite(
        str(
            sample_folder
            / "water_probability.jpg"
        ),
        probability_image
    )

    # --------------------------------
    # HIGH + LOW MASKS
    # --------------------------------

    high_mask = (
        water_probability
        >= HIGH_THRESHOLD
    ).astype(
        np.uint8
    ) * 255

    low_mask = (
        water_probability
        >= LOW_THRESHOLD
    ).astype(
        np.uint8
    ) * 255

    kernel = np.ones(
        (5, 5),
        np.uint8
    )

    high_mask = cv2.morphologyEx(
        high_mask,
        cv2.MORPH_OPEN,
        kernel
    )

    high_mask = cv2.morphologyEx(
        high_mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    low_mask = cv2.morphologyEx(
        low_mask,
        cv2.MORPH_OPEN,
        kernel
    )

    low_mask = cv2.morphologyEx(
        low_mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    # --------------------------------
    # FIND BORDER-TOUCHING SEEDS
    # --------------------------------

    num_labels, labels, stats, _ = (
        cv2.connectedComponentsWithStats(
            high_mask,
            connectivity=8
        )
    )

    seed_mask = np.zeros_like(
        high_mask
    )

    for label_id in range(
        1,
        num_labels
    ):

        area = stats[
            label_id,
            cv2.CC_STAT_AREA
        ]

        component = np.zeros_like(
            high_mask
        )

        component[
            labels == label_id
        ] = 255

        if (
            area
            >= MIN_COMPONENT_AREA
            and touches_border(
                component
            )
        ):
            seed_mask[
                labels == label_id
            ] = 255

    cv2.imwrite(
        str(
            sample_folder
            / "water_seed_mask.jpg"
        ),
        seed_mask
    )

    # --------------------------------
    # GROW THROUGH LOW-CONFIDENCE WATER
    # --------------------------------

    num_labels, labels, stats, _ = (
        cv2.connectedComponentsWithStats(
            low_mask,
            connectivity=8
        )
    )

    grown_mask = np.zeros_like(
        low_mask
    )

    for label_id in range(
        1,
        num_labels
    ):

        area = stats[
            label_id,
            cv2.CC_STAT_AREA
        ]

        component_region = (
            labels == label_id
        )

        overlap = np.any(
            seed_mask[
                component_region
            ] > 0
        )

        if (
            area
            >= MIN_COMPONENT_AREA
            and overlap
        ):
            grown_mask[
                component_region
            ] = 255

    merge_kernel = np.ones(
        (13, 13),
        np.uint8
    )

    grown_mask = cv2.morphologyEx(
        grown_mask,
        cv2.MORPH_CLOSE,
        merge_kernel
    )

    grown_mask = cv2.medianBlur(
        grown_mask,
        5
    )

    # --------------------------------
    # DRAW SHORELINE
    # --------------------------------

    contours, _ = cv2.findContours(
        grown_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    original = cv2.imread(
        str(image_path)
    )

    result = original.copy()

    cv2.drawContours(
        result,
        contours,
        -1,
        (0, 0, 255),
        3
    )

    cv2.imwrite(
        str(
            sample_folder
            / "water_mask.jpg"
        ),
        grown_mask
    )

    cv2.imwrite(
        str(
            sample_folder
            / "shoreline.jpg"
        ),
        result
    )


# ============================================================
# MAIN
# ============================================================

print()
print(
    "VIMS NEARMAP MULTI-SAMPLE TEST"
)
print(
    "=============================="
)

OUTPUT_ROOT.mkdir(
    exist_ok=True
)

locations = (
    get_shoreline_locations()
)


# --------------------------------
# LOAD AI MODEL ONCE
# --------------------------------

print()
print("Loading SegFormer model...")

processor = (
    AutoImageProcessor
    .from_pretrained(
        MODEL_NAME
    )
)

model = (
    SegformerForSemanticSegmentation
    .from_pretrained(
        MODEL_NAME
    )
)

model.eval()

print("Model loaded.")


# --------------------------------
# WATER LABEL IDS
# --------------------------------

water_ids = []

for class_id, label in (
    model.config.id2label.items()
):

    if (
        label.strip().lower()
        in WATER_LABELS
    ):
        water_ids.append(
            int(class_id)
        )

print(
    "Water IDs:",
    water_ids
)


# --------------------------------
# PROCESS SAMPLES
# --------------------------------

for number, (
    latitude,
    longitude
) in enumerate(
    locations,
    start=1
):

    print()
    print(
        f"========== SAMPLE "
        f"{number}/{NUM_SAMPLES} "
        f"=========="
    )

    print(
        f"Latitude:  "
        f"{latitude}"
    )

    print(
        f"Longitude: "
        f"{longitude}"
    )

    sample_folder = (
        OUTPUT_ROOT
        / f"sample_{number:02d}"
    )

    sample_folder.mkdir(
        exist_ok=True
    )

    original_file = (
        sample_folder
        / "original.jpg"
    )

    print(
        "Downloading "
        "Nearmap imagery..."
    )

    download_nearmap_image(
        latitude,
        longitude,
        original_file
    )

    print(
        "Running shoreline "
        "segmentation..."
    )

    segment_image(
        original_file,
        sample_folder,
        processor,
        model,
        water_ids
    )

    metadata = {
        "sample": number,
        "latitude": latitude,
        "longitude": longitude,
        "zoom": ZOOM,
        "high_threshold": (
            HIGH_THRESHOLD
        ),
        "low_threshold": (
            LOW_THRESHOLD
        )
    }

    with open(
        sample_folder
        / "metadata.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            metadata,
            file,
            indent=2
        )

    print(
        f"Finished sample "
        f"{number}."
    )


print()
print(
    "=============================="
)
print("ALL SAMPLES COMPLETE")
print(
    f"Results saved in:"
)
print(OUTPUT_ROOT)
print(
    "=============================="
)