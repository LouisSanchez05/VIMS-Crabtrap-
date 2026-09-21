import os
import csv
import json
import time
import base64
import mimetypes
import sqlite3
import re
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


# ============================================================
# SETTINGS
# ============================================================

BASE_FOLDER = Path(
    r"\\ccrmspace\ccrm\Projects\National derelict TRAP program\Virginia\photos\2026"
)

DATABASE_FILE = Path("production_results_with_year.db")
CSV_FILE = Path("production_results_with_year.csv")

BATCH_START = 2
BATCH_END = 29

CSV_EXPORT_INTERVAL = 100

MODEL = "gpt-5.6-terra"

MAX_RETRIES = 3


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".tif",
    ".tiff",
}


# 9 color classes + wire + unknown = 11 total outputs.
#
# Brown is intentionally not included because mud/rust is
# commonly brown and could cause false classifications.
# "wire" represents a bare/uncoated metal trap, including rusty wire.
ALLOWED_COLORS = [
    "green",
    "black",
    "blue",
    "red",
    "orange",
    "yellow",
    "white",
    "gray",
    "purple",
    "wire",
    "unknown",
]

CONFIDENCE_LEVELS = [
    "low",
    "medium",
    "high",
]


# ============================================================
# OPENAI
# ============================================================

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise RuntimeError(
        "OPENAI_API_KEY not found. "
        "Make sure your .env file exists."
    )

client = OpenAI(api_key=api_key)


# ============================================================
# DATABASE
# ============================================================

def initialize_database():
    connection = sqlite3.connect(DATABASE_FILE)

    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS results (
            relative_path TEXT PRIMARY KEY,
            batch TEXT NOT NULL,
            filename TEXT NOT NULL,
            first_name TEXT,
            last_name TEXT,
            year INTEGER NOT NULL DEFAULT 9999,
            color TEXT,
            confidence TEXT,
            status TEXT NOT NULL,
            error TEXT,
            processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )

    connection.commit()

    return connection


def save_result(
    connection,
    relative_path,
    batch,
    filename,
    first_name,
    last_name,
    year,
    color,
    confidence,
    status,
    error=None,
):
    connection.execute(
        """
        INSERT OR REPLACE INTO results (
            relative_path,
            batch,
            filename,
            first_name,
            last_name,
            year,
            color,
            confidence,
            status,
            error,
            processed_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        """,
        (
            relative_path,
            batch,
            filename,
            first_name,
            last_name,
            year,
            color,
            confidence,
            status,
            error,
        ),
    )

    # Save every result immediately.
    connection.commit()

def already_successful(connection, relative_path):
    result = connection.execute(
        """
        SELECT status
        FROM results
        WHERE relative_path = ?
        """,
        (relative_path,),
    ).fetchone()

    return result is not None and result[0] == "success"


# ============================================================
# CSV
# ============================================================
def export_csv(connection):
    rows = connection.execute(
        """
        SELECT
            relative_path,
            batch,
            filename,
            first_name,
            last_name,
            year,
            color,
            confidence,
            status,
            error
        FROM results
        ORDER BY batch, filename
        """
    ).fetchall()

    temp_file = Path(str(CSV_FILE) + ".tmp")

    with open(
        temp_file,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.writer(file)

        writer.writerow(
            [
                "relative_path",
                "batch",
                "filename",
                "first_name",
                "last_name",
                "year",
                "color",
                "confidence",
                "status",
                "error",
            ]
        )

        writer.writerows(rows)

        file.flush()
        os.fsync(file.fileno())

    os.replace(temp_file, CSV_FILE)

# ============================================================
# FILENAME
# ============================================================

def extract_name(filename):
    """
    Example:

    a_pruitt_2012_01.JPG

    becomes:

    A
    Pruitt
    """

    stem = Path(filename).stem
    parts = stem.split("_")

    if len(parts) < 2:
        return "UNKNOWN", "UNKNOWN"

    first_name = parts[0].strip().title()
    last_name = parts[1].strip().title()

    return first_name, last_name


def extract_year(filename):
    """
    Extract a plausible collection year from the filename.

    Only years 2007 through 2014 are accepted.
    Anything else (including 2001 or no year) becomes 9999.
    """

    stem = Path(filename).stem
    candidates = re.findall(r"(?<!\d)(\d{4})(?!\d)", stem)

    for candidate in candidates:
        year = int(candidate)
        if 2007 <= year <= 2014:
            return year

    return 9999


# ============================================================
# IMAGE ENCODING
# ============================================================

def image_to_data_url(image_path):

    mime_type, _ = mimetypes.guess_type(image_path)

    if mime_type is None:
        mime_type = "image/jpeg"

    with open(image_path, "rb") as image_file:
        encoded_image = base64.b64encode(
            image_file.read()
        ).decode("utf-8")

    return (
        f"data:{mime_type};base64,"
        f"{encoded_image}"
    )


# ============================================================
# AI CLASSIFICATION
# ============================================================

def classify_trap(image_path):

    image_data = image_to_data_url(image_path)

    allowed = ", ".join(ALLOWED_COLORS)

    prompt = f"""
You are analyzing a photograph of a recovered derelict crab trap.

There is exactly ONE crab trap in the photograph.

Your task is to determine the ORIGINAL COLOR of the crab trap's
wire or plastic coating.

You may only choose one of these values:

{allowed}

The trap may be heavily covered by:
- mud
- rust
- sediment
- algae
- marine growth
- barnacles
- debris

IMPORTANT:

Analyze ONLY the crab trap itself.

Ignore colors belonging to:
- people's clothing
- orange waterproof bibs
- gloves
- boats
- decks
- ropes
- floats
- water
- objects caught inside the trap
- mud
- rust
- algae
- marine organisms
- debris

Look specifically for exposed sections of wire mesh, frame,
or plastic-coated wire.

Do NOT simply choose the most common color in the photograph.

Do NOT interpret brown mud or brown/orange rust as the
original trap color.

WIRE CLASS:
Return "wire" when the crab trap appears to be bare/uncoated metal wire
rather than plastic-coated colored wire. A wire trap may be gray, metallic,
darkened, or heavily rusted.

Rust does NOT make a trap orange, red, brown, or black.

Use "black" only when there is visible evidence of an actual black coating
on the trap wire or frame. If the trap is rusty/dark bare metal with no
clear colored plastic coating, classify it as "wire".

CONFIDENCE:

high:
The underlying trap coating is clearly visible in multiple areas
and consistently shows the same color.

medium:
There is meaningful visual evidence for the color, but the trap
is partially obscured or the visible coating is limited.

low:
There is very little visible coating, the evidence conflicts,
or the color is difficult to determine.

If the original coating cannot reasonably be determined,
return:

color: unknown
confidence: low

Do not force a guess.

Return JSON only in exactly this format:

{{
    "color": "green",
    "confidence": "high"
}}
"""

    response = client.responses.create(
        model=MODEL,

        input=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": prompt,
                    },
                    {
                        "type": "input_image",
                        "image_url": image_data,
                        "detail": "high",
                    },
                ],
            }
        ],
    )

    text = response.output_text.strip()

    # Occasionally models wrap JSON in markdown fences.
    text = text.replace("```json", "")
    text = text.replace("```", "")
    text = text.strip()

    result = json.loads(text)

    color = result.get("color", "").lower()
    confidence = result.get("confidence", "").lower()

    # Validate the returned values ourselves.
    if color not in ALLOWED_COLORS:
        raise ValueError(
            f"Invalid color returned: {color}"
        )

    if confidence not in CONFIDENCE_LEVELS:
        raise ValueError(
            f"Invalid confidence returned: {confidence}"
        )

    return color, confidence


# ============================================================
# RETRIES
# ============================================================

def classify_with_retries(image_path):

    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):

        try:
            return classify_trap(image_path)

        except Exception as error:

            last_error = error

            print(
                f"    Attempt {attempt}/{MAX_RETRIES} failed"
            )

            print(f"    {error}")

            if attempt < MAX_RETRIES:

                wait_seconds = 2 ** attempt

                print(
                    f"    Retrying in "
                    f"{wait_seconds} seconds..."
                )

                time.sleep(wait_seconds)

    raise last_error


# ============================================================
# USER MENU
# ============================================================

def choose_processing_mode():
    """
    Returns:
        False -> Resume mode: skip rows already marked successful.
        True  -> Override mode: reprocess selected images and replace results.
        None  -> Exit.
    """

    while True:
        print()
        print("=" * 60)
        print("CRAB TRAP PROCESSING TOOL")
        print("=" * 60)
        print("Processing mode:")
        print("  1. Resume / skip completed images")
        print("  2. Override / reprocess existing images")
        print("  3. Exit")

        choice = input("Select an option (1-3): ").strip()

        if choice == "1":
            return False

        if choice == "2":
            print()
            print("WARNING:")
            print("Override mode will reprocess images even if they")
            print("already have successful results in the database.")
            print("New successful results will replace the old results.")
            print()

            confirm = input(
                "Continue with override mode? (y/n): "
            ).strip().lower()

            if confirm in {"y", "yes"}:
                return True

            print("Override cancelled.")
            continue

        if choice == "3":
            return None

        print("Invalid choice. Please enter 1, 2, or 3.")


def choose_batches():
    """
    Returns a list of batch numbers to scan, or None if the user exits.
    """

    while True:
        print()
        print("Batch selection:")
        print("  1. Run one batch")
        print(
            f"  2. Run all batches "
            f"({BATCH_START:02d}-{BATCH_END:02d})"
        )
        print("  3. Exit")

        choice = input("Select an option (1-3): ").strip()

        if choice == "1":
            while True:
                value = input(
                    f"Enter batch number "
                    f"({BATCH_START:02d}-{BATCH_END:02d}): "
                ).strip()

                try:
                    batch_number = int(value)
                except ValueError:
                    print("Please enter a valid batch number.")
                    continue

                if BATCH_START <= batch_number <= BATCH_END:
                    return [batch_number]

                print(
                    f"Batch must be between "
                    f"{BATCH_START:02d} and {BATCH_END:02d}."
                )

        elif choice == "2":
            return list(range(BATCH_START, BATCH_END + 1))

        elif choice == "3":
            return None

        else:
            print("Invalid choice. Please enter 1, 2, or 3.")


# ============================================================
# MAIN
# ============================================================
def get_batch_images(batch_folder, retries=5):

    for attempt in range(1, retries + 1):

        try:
            images = []

            for file in batch_folder.iterdir():
                if file.suffix.lower() in IMAGE_EXTENSIONS:
                    images.append(file)

            return sorted(images)

        except OSError as error:

            print()
            print(
                f"Network error while reading "
                f"{batch_folder.name}"
            )
            print(error)

            if attempt == retries:
                raise

            wait_seconds = 5 * attempt

            print(
                f"Retrying in {wait_seconds} seconds..."
            )

            time.sleep(wait_seconds)
def main():

    if not BASE_FOLDER.exists():
        print("Cannot access:")
        print(BASE_FOLDER)
        print()
        print("Make sure Connect Tunnel VPN is connected.")
        return

    override_existing = choose_processing_mode()

    if override_existing is None:
        print("Exiting.")
        return

    selected_batches = choose_batches()

    if selected_batches is None:
        print("Exiting.")
        return

    print()
    print("=" * 60)
    print("RUN SETTINGS")
    print("=" * 60)

    if override_existing:
        print("Mode: OVERRIDE / REPROCESS")
        print("Existing successful images: REPROCESS")
    else:
        print("Mode: RESUME")
        print("Existing successful images: SKIP")

    if len(selected_batches) == 1:
        print(f"Batch: batch_{selected_batches[0]:02d}")
    else:
        print(
            f"Batches: batch_{BATCH_START:02d} "
            f"through batch_{BATCH_END:02d}"
        )

    print(f"Database: {DATABASE_FILE}")
    print(f"CSV: {CSV_FILE}")
    print("=" * 60)
    print()

    connection = initialize_database()

    images = []

    print("Scanning batch folders...")
    print("=" * 60)

    for batch_number in selected_batches:

        batch_name = f"batch_{batch_number:02d}"
        batch_folder = BASE_FOLDER / batch_name

        if not batch_folder.exists():
            print(f"{batch_name}: MISSING - skipping")
            continue

        try:
            batch_images = get_batch_images(batch_folder)

        except OSError:
            print()
            print(f"Could not reliably access {batch_name}.")
            print("Reconnect VPN and run the script again.")
            connection.close()
            return

        print(f"{batch_name}: {len(batch_images)} images")

        for image_path in batch_images:
            images.append(
                (
                    batch_name,
                    image_path,
                )
            )

    total = len(images)

    print("=" * 60)
    print(f"TOTAL IMAGES FOUND: {total}")
    print("=" * 60)
    print()

    processed_this_run = 0

    try:

        for number, (batch_name, image_path) in enumerate(
            images,
            start=1,
        ):

            if not BASE_FOLDER.exists():
                print()
                print("NETWORK SHARE DISCONNECTED.")
                print("Stopping safely.")
                break

            filename = image_path.name

            relative_path = str(
                image_path.relative_to(BASE_FOLDER)
            )

            if (
                not override_existing
                and already_successful(
                    connection,
                    relative_path,
                )
            ):

                print(
                    f"[{number}/{total}] "
                    f"Skipping {relative_path} "
                    f"(already complete)"
                )

                continue

            first_name, last_name = extract_name(
                filename
            )
            year = extract_year(filename)

            print(
                f"[{number}/{total}] "
                f"Processing {relative_path}"
            )

            try:

                color, confidence = (
                    classify_with_retries(
                        image_path
                    )
                )

                save_result(
                    connection=connection,
                    relative_path=relative_path,
                    batch=batch_name,
                    filename=filename,
                    first_name=first_name,
                    last_name=last_name,
                    year=year,
                    color=color,
                    confidence=confidence,
                    status="success",
                )

                print(
                    f"    {first_name} {last_name}"
                )
                print(
                    f"    Year: {year}"
                )
                print(
                    f"    Color: {color}"
                )
                print(
                    f"    Confidence: {confidence}"
                )

            except Exception as error:

                error_message = str(error)[:1000]

                save_result(
                    connection=connection,
                    relative_path=relative_path,
                    batch=batch_name,
                    filename=filename,
                    first_name=first_name,
                    last_name=last_name,
                    year=year,
                    color=None,
                    confidence=None,
                    status="failed",
                    error=error_message,
                )

                print("    FAILED")
                print(f"    {error_message}")

            processed_this_run += 1

            if (
                processed_this_run
                % CSV_EXPORT_INTERVAL
                == 0
            ):
                print("    Updating CSV...")
                export_csv(connection)

            print()

    except KeyboardInterrupt:

        print()
        print("Stopped by user.")
        print(
            "Completed results have already been saved."
        )

    finally:

        print("Creating final CSV...")
        export_csv(connection)
        connection.close()

    print()
    print("=" * 60)
    print("RUN FINISHED")
    print(f"Database: {DATABASE_FILE}")
    print(f"CSV: {CSV_FILE}")


if __name__ == "__main__":
    main()