# backend/pipeline/step2_spatial_layers.py
import os, json
import numpy as np
import pandas as pd

OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data"))

DELHI_EMISSION_SOURCES = [
    {"source_id": 1, "name": "Bawana Industrial Estate", "type": "Heavy Industrial", "lat": 28.7720, "lon": 77.0398, "intensity": 850},
    {"source_id": 2, "name": "Narela Industrial Estate", "type": "Heavy Industrial", "lat": 28.8540, "lon": 77.0940, "intensity": 720},
    {"source_id": 3, "name": "Wazirpur Industrial Area", "type": "Industrial Stack", "lat": 28.7060, "lon": 77.1700, "intensity": 680},
    {"source_id": 4, "name": "Okhla Industrial Area", "type": "Industrial Stack", "lat": 28.5355, "lon": 77.2700, "intensity": 590},
    {"source_id": 5, "name": "Jhilmil Industrial Area", "type": "Industrial Stack", "lat": 28.6700, "lon": 77.3100, "intensity": 520},
    {"source_id": 6, "name": "Mayapuri Industrial Area", "type": "Industrial Stack", "lat": 28.6335, "lon": 77.1098, "intensity": 460},
    {"source_id": 7, "name": "Delhi Metro Phase IV", "type": "Construction", "lat": 28.6800, "lon": 77.2200, "intensity": 380},
    {"source_id": 8, "name": "Pragati Maidan Redevelopment", "type": "Construction", "lat": 28.6192, "lon": 77.2414, "intensity": 290},
]

GRID_CENTER_LAT, GRID_CENTER_LON, GRID_SIZE = 28.6280, 77.2090, 45
DEG_PER_KM_LAT, DEG_PER_KM_LON = 0.009, 0.0105

def build_grid():
    records = []
    for cell_id in range(GRID_SIZE * GRID_SIZE):
        x, y = cell_id % GRID_SIZE, cell_id // GRID_SIZE
        lat = GRID_CENTER_LAT + (y - GRID_SIZE // 2) * DEG_PER_KM_LAT
        lon = GRID_CENTER_LON + (x - GRID_SIZE // 2) * DEG_PER_KM_LON
        records.append({"cell_id": cell_id, "x": x, "y": y, "lat": round(lat, 6), "lon": round(lon, 6)})
    return pd.DataFrame(records)

def compute_road_density_osm(grid_df: pd.DataFrame) -> pd.DataFrame:
    try:
        import osmnx as ox
        north, south = GRID_CENTER_LAT + (GRID_SIZE/2)*DEG_PER_KM_LAT, GRID_CENTER_LAT - (GRID_SIZE/2)*DEG_PER_KM_LAT
        east, west = GRID_CENTER_LON + (GRID_SIZE/2)*DEG_PER_KM_LON, GRID_CENTER_LON - (GRID_SIZE/2)*DEG_PER_KM_LON
        G = ox.graph_from_bbox(bbox=(north, south, east, west), network_type="drive", custom_filter='["highway"~"motorway|trunk|primary|secondary|tertiary|residential"]')
        edges = ox.graph_to_gdfs(G, nodes=False)
    except Exception:
        grid_df["road_density_m"] = 500.0
        return grid_df

    from shapely.geometry import box as shapely_box
    road_densities = []
    for _, cell in grid_df.iterrows():
        cell_box = shapely_box(cell["lon"] - DEG_PER_KM_LON/2, cell["lat"] - DEG_PER_KM_LAT/2, cell["lon"] + DEG_PER_KM_LON/2, cell["lat"] + DEG_PER_KM_LAT/2)
        try: road_densities.append(edges[edges.geometry.intersects(cell_box)].geometry.intersection(cell_box).length.sum() * 111_000)
        except Exception: road_densities.append(300.0)
    grid_df["road_density_m"] = road_densities
    return grid_df

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, "delhi_emission_sources.json"), "w") as f: json.dump(DELHI_EMISSION_SOURCES, f, indent=2)
    grid_df = compute_road_density_osm(build_grid())
    grid_df.to_csv(os.path.join(OUTPUT_DIR, "delhi_grid_road_density.csv"), index=False)

if __name__ == "__main__": main()