from pathlib import Path

readme = """# VIMS Crab Trap Image Classifier

Python tool for processing VIMS/CCRM crab trap images and generating a CSV/database containing photographer name, image year, trap color, confidence, and processing status.

## What it does

- Reads images directly from the VIMS/CCRM network share.
- Extracts first name, last name, and year from the filename.
- Uses `9999` when a valid year cannot be determined.
- Classifies trap color using the OpenAI API.
- Supports the `wire` and `unknown` classes.
- Saves results to SQLite after each image.
- Periodically exports results to CSV.
- Can resume a previous run without reprocessing completed images.
- Can optionally override and reprocess existing successful results.
- Can process one batch or all available batches.

## Requirements

- Python 3.11+ recommended
- Access to the VIMS/CCRM network share
- VIMS VPN/network access if required
- OpenAI API key

Install dependencies:

```cmd
py -m pip install -r requirements.txt
