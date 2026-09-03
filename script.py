import os
import csv
import json
import time
import base64
import mimetypes
import sqlite3
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


# ============================================================
# SETTINGS
# ============================================================

IMAGE_FOLDER = Path(
    r"\\ccrmspace\ccrm\Projects\National derelict TRAP program\Virginia\photos\2026\batch_02"
)

DATABASE_FILE = Path("results.db")
CSV_FILE = Path("results.csv")

TEST_LIMIT = 20

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


# 9 possible colors + unknown = 10 total outputs.
#
# Brown is intentionally not included because mud/rust is
# commonly brown and could cause false classifications.
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
            filename TEXT PRIMARY KEY,
            first_name TEXT,
            last_name TEXT,
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
    filename,
    first_name,
    last_name,
    color,
    confidence,
    status,
    error=None,
):
    connection.execute(
        """
        INSERT OR REPLACE INTO results (
            filename,
            first_name,
            last_name,
            color,
            confidence,
            status,
            error,
            processed_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        """,
        (
            filename,
            first_name,
            last_name,
            color,
            confidence,
            status,
            error,
        ),
    )

    # Save THIS result permanently before continuing.
    connection.commit()


def already_successful(connection, filename):
    result = connection.execute(
        """
        SELECT status
        FROM results
        WHERE filename = ?
        """,
        (filename,),
    ).fetchone()

    return result is not None and result[0] == "success"


# ============================================================
# CSV
# ============================================================

def export_csv(connection):
    rows = connection.execute(
        """
        SELECT
            filename,
            first_name,
            last_name,
            color,
            confidence,
            status,
            error
        FROM results
        ORDER BY filename
        """
    ).fetchall()

    temp_file = Path("results.csv.tmp")

    with open(
        temp_file,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.writer(file)

        writer.writerow(
            [
                "filename",
                "first_name",
                "last_name",
                "color",
                "confidence",
                "status",
                "error",
            ]
        )

        writer.writerows(rows)

        file.flush()
        os.fsync(file.fileno())

    # Replace old CSV only after new one is fully written.
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
# MAIN
# ============================================================

def main():

    if not IMAGE_FOLDER.exists():

        print("Cannot access network folder:")
        print(IMAGE_FOLDER)
        print()
        print("Check your VPN connection.")

        return

    connection = initialize_database()

    images = sorted(
        file
        for file in IMAGE_FOLDER.iterdir()
        if (
            file.is_file()
            and file.suffix.lower()
            in IMAGE_EXTENSIONS
        )
    )

    print(f"Found {len(images)} images.")

    # Only process first TEST_LIMIT images.
    images = images[:TEST_LIMIT]

    total = len(images)

    print(
        f"Processing first {total} images."
    )

    print("=" * 60)

    try:

        for number, image_path in enumerate(
            images,
            start=1,
        ):

            filename = image_path.name

            # ----------------------------------------
            # Check network share before processing.
            # ----------------------------------------

            if not IMAGE_FOLDER.exists():

                print()
                print(
                    "NETWORK SHARE DISCONNECTED."
                )

                print(
                    "Stopping safely. "
                    "Existing results are preserved."
                )

                break

            # ----------------------------------------
            # Resume / checkpoint
            # ----------------------------------------

            if already_successful(
                connection,
                filename,
            ):

                print(
                    f"[{number}/{total}] "
                    f"Skipping {filename} "
                    f"(already complete)"
                )

                continue

            first_name, last_name = extract_name(
                filename
            )

            print(
                f"[{number}/{total}] "
                f"Processing {filename}"
            )

            try:

                color, confidence = (
                    classify_with_retries(
                        image_path
                    )
                )

                save_result(
                    connection=connection,
                    filename=filename,
                    first_name=first_name,
                    last_name=last_name,
                    color=color,
                    confidence=confidence,
                    status="success",
                )

                print(
                    f"    {first_name} "
                    f"{last_name}"
                )

                print(
                    f"    Color: {color}"
                )

                print(
                    f"    Confidence: "
                    f"{confidence}"
                )

            except Exception as error:

                error_message = str(error)[:1000]

                save_result(
                    connection=connection,
                    filename=filename,
                    first_name=first_name,
                    last_name=last_name,
                    color=None,
                    confidence=None,
                    status="failed",
                    error=error_message,
                )

                print("    FAILED")
                print(
                    f"    {error_message}"
                )

            # Export after every image for this small test.
            export_csv(connection)

            print()

    except KeyboardInterrupt:

        print()
        print("=" * 60)
        print("Processing stopped by user.")
        print(
            "Everything completed so far "
            "has been saved."
        )

    finally:

        # Rebuild CSV from database before exit.
        export_csv(connection)

        connection.close()

    print()
    print("=" * 60)
    print("Finished.")
    print(f"Results: {CSV_FILE}")
    print(
        f"Checkpoint database: "
        f"{DATABASE_FILE}"
    )


if __name__ == "__main__":
    main()