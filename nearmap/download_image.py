import os
import math
import requests
from io import BytesIO
from PIL import Image
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("NEARMAP_API_KEY")

if not API_KEY:
    raise ValueError("NEARMAP_API_KEY not found in .env")

TILE_SIZE = 256
GRID_SIZE = 3


def lat_lon_to_tile(lat, lon, zoom):
    lat_rad = math.radians(lat)
    n = 2 ** zoom

    x = int((lon + 180.0) / 360.0 * n)

    y = int(
        (
            1.0
            - math.asinh(math.tan(lat_rad)) / math.pi
        )
        / 2.0
        * n
    )

    return x, y


# -------------------------
# USER INPUT
# -------------------------

print("\nNEARMAP IMAGE DOWNLOADER")
print("------------------------")

latitude = float(input("Enter latitude: "))
longitude = float(input("Enter longitude: "))

zoom_input = input("Enter zoom [19]: ").strip()

if zoom_input == "":
    zoom = 19
else:
    zoom = int(zoom_input)


# -------------------------
# FIND CENTER TILE
# -------------------------

center_x, center_y = lat_lon_to_tile(
    latitude,
    longitude,
    zoom
)

print()
print(f"Location: {latitude}, {longitude}")
print(f"Zoom: {zoom}")
print(f"Center tile: x={center_x}, y={center_y}")
print()
print("Downloading 3x3 Nearmap imagery...")


# -------------------------
# CREATE FINAL IMAGE
# -------------------------

final_image = Image.new(
    "RGB",
    (
        TILE_SIZE * GRID_SIZE,
        TILE_SIZE * GRID_SIZE
    )
)


# -------------------------
# DOWNLOAD 9 TILES
# -------------------------

for row, dy in enumerate([-1, 0, 1]):

    for col, dx in enumerate([-1, 0, 1]):

        x = center_x + dx
        y = center_y + dy

        print(f"Downloading tile x={x}, y={y}")

        url = (
            f"https://us0.nearmap.com/maps/"
            f"z={zoom}"
            f"&x={x}"
            f"&y={y}"
            f"&version=2"
            f"&nml=Vert"
            f"&client=vims_nearmap"
            f"&httpauth=false"
            f"&apikey={API_KEY}"
        )

        response = requests.get(
            url,
            timeout=30
        )

        content_type = response.headers.get(
            "Content-Type",
            ""
        )

        if (
            response.status_code != 200
            or not content_type.startswith("image/")
        ):
            print()
            print("ERROR downloading tile.")
            print("Status:", response.status_code)
            print("Content-Type:", content_type)

            if not content_type.startswith("image/"):
                print(response.text[:500])

            raise SystemExit

        tile = Image.open(
            BytesIO(response.content)
        ).convert("RGB")

        final_image.paste(
            tile,
            (
                col * TILE_SIZE,
                row * TILE_SIZE
            )
        )


# -------------------------
# SAVE IMAGE
# -------------------------

filename = "nearmap_test.jpg"

final_image.save(
    filename,
    "JPEG",
    quality=95
)

print()
print("------------------------")
print("SUCCESS!")
print(f"Saved as: {filename}")
print(
    f"Image size: "
    f"{final_image.width}x{final_image.height}"
)
print("------------------------")