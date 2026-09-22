import requests
import json

# VIMS 2017 Bay and Ocean Side Shoreline
URL = (
    "https://mobjack.vims.edu/ArcGIS/rest/services/"
    "VIMS_SSP/All_WebShoreline1937_1949_2009_2017/"
    "FeatureServer/3/query"
)

params = {
    "where": "1=1",
    "outFields": "*",
    "returnGeometry": "true",
    "outSR": "4326",       # latitude / longitude
    "resultRecordCount": 10,
    "f": "geojson"
}

print()
print("VIMS SHORELINE DOWNLOADER")
print("-------------------------")
print("Downloading shoreline data...")

response = requests.get(
    URL,
    params=params,
    timeout=30
)

response.raise_for_status()

data = response.json()

features = data.get("features", [])

print(f"Downloaded {len(features)} shoreline features.")

if not features:
    print("No shoreline features returned.")
    raise SystemExit


# Save the raw shoreline GeoJSON
with open(
    "vims_coastline_test.geojson",
    "w",
    encoding="utf-8"
) as file:
    json.dump(data, file, indent=2)


# Find a usable coordinate from the shoreline
sample_coordinates = []

for feature in features:

    geometry = feature.get("geometry")

    if not geometry:
        continue

    geometry_type = geometry.get("type")
    coordinates = geometry.get("coordinates", [])

    if geometry_type == "LineString":

        for lon, lat, *rest in coordinates:
            sample_coordinates.append(
                (lat, lon)
            )

    elif geometry_type == "MultiLineString":

        for line in coordinates:

            for lon, lat, *rest in line:
                sample_coordinates.append(
                    (lat, lon)
                )


print()
print(f"Found {len(sample_coordinates)} shoreline points.")

if sample_coordinates:

    # Pick approximately the middle coordinate
    middle = len(sample_coordinates) // 2

    latitude, longitude = sample_coordinates[middle]

    print()
    print("Sample shoreline coordinate:")
    print(f"Latitude:  {latitude}")
    print(f"Longitude: {longitude}")

    print()
    print("You can test this coordinate")
    print("with download_image.py.")


print()
print("Saved:")
print("vims_coastline_test.geojson")
print()
print("SUCCESS!")