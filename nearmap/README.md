# VIMS Nearmap Shoreline Prototype

Prototype workflow for using Nearmap aerial imagery to identify Virginia shoreline locations and test automated shoreline detection.

## Current Workflow

- Pull shoreline coordinates from a VIMS ArcGIS shoreline dataset.
- Download Nearmap aerial imagery using the Nearmap API.
- Stitch 3x3 image tiles into a 768x768 image.
- Run a pretrained SegFormer semantic-segmentation model.
- Create water masks and candidate shoreline boundaries.
- Test the same detection workflow across multiple shoreline samples for manual comparison.

## Main Scripts

- `download_image.py` — downloads a 3x3 Nearmap image for a latitude/longitude.
- `get_coastline.py` — retrieves shoreline locations from the VIMS shoreline dataset.
- `segment_shoreline.py` — creates a water mask and shoreline detection from a Nearmap image.
- `sample_run/run_samples.py` — automatically tests multiple shoreline locations and saves results for review.

## Setup

Install dependencies:

```bash
py -m pip install requests python-dotenv pillow opencv-python numpy torch torchvision transformers
```

Create a `.env` file:

```text
NEARMAP_API_KEY=your_api_key_here
```

Do not commit `.env` or API keys to GitHub.

## Current Goal

The current goal is to determine how reliably the model can identify the water/land boundary across different Nearmap coastal scenes before exporting shoreline results for GIS use.
