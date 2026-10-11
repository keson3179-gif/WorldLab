"""Fixed places and travel distances for World v0.1.

Place ids stay in Python metadata. The utility prompt receives attributes only.
"""

from __future__ import annotations

from typing import Any

CROWDING_FORMULA = "min(1.0, occupancy / capacity)"
PLACE_IDS = ("site_01", "site_02", "site_03", "site_04", "site_05")
PLACES: tuple[dict[str, Any], ...] = (
    {"place_id": "site_01", "visual_attractiveness": 0.82, "novelty": 0.24, "capacity": 8},
    {"place_id": "site_02", "visual_attractiveness": 0.67, "novelty": 0.51, "capacity": 10},
    {"place_id": "site_03", "visual_attractiveness": 0.48, "novelty": 0.78, "capacity": 12},
    {"place_id": "site_04", "visual_attractiveness": 0.33, "novelty": 0.63, "capacity": 16},
    {"place_id": "site_05", "visual_attractiveness": 0.16, "novelty": 0.12, "capacity": 20},
)
_DISTANCE_ROWS = (
    (0.0, 0.2, 0.4, 0.6, 0.8),
    (0.2, 0.0, 0.25, 0.5, 0.7),
    (0.4, 0.25, 0.0, 0.3, 0.55),
    (0.6, 0.5, 0.3, 0.0, 0.35),
    (0.8, 0.7, 0.55, 0.35, 0.0),
)


def place_index() -> dict[str, dict[str, Any]]:
    return {str(place["place_id"]): place for place in PLACES}


def distance_matrix() -> dict[str, dict[str, float]]:
    matrix = {
        origin: {destination: float(_DISTANCE_ROWS[row][column]) for column, destination in enumerate(PLACE_IDS)}
        for row, origin in enumerate(PLACE_IDS)
    }
    validate_distances(matrix)
    return matrix


def validate_distances(matrix: dict[str, dict[str, float]]) -> None:
    for origin in PLACE_IDS:
        for destination in PLACE_IDS:
            value = float(matrix[origin][destination])
            if value < 0.0 or value > 1.0:
                raise ValueError("distance must be within 0 and 1")
            if origin == destination and value != 0.0:
                raise ValueError("stay distance must be 0")
            if abs(value - float(matrix[destination][origin])) > 1e-12:
                raise ValueError("distance matrix must be symmetric")


def travel_distance(matrix: dict[str, dict[str, float]], origin: str, destination: str) -> float:
    if origin not in matrix or destination not in matrix[origin]:
        raise ValueError("unknown place")
    return float(matrix[origin][destination])


def crowding(occupancy: int, capacity: int) -> float:
    if capacity <= 0:
        raise ValueError("capacity must be positive")
    if occupancy < 0:
        raise ValueError("occupancy cannot be negative")
    return min(1.0, occupancy / capacity)


def occupancy_of(locations: dict[str, str]) -> dict[str, int]:
    counts = {place_id: 0 for place_id in PLACE_IDS}
    for place_id in locations.values():
        if place_id not in counts:
            raise ValueError(f"invalid place: {place_id}")
        counts[place_id] += 1
    return counts


def crowding_of(occupancy: dict[str, int]) -> dict[str, float]:
    places = place_index()
    return {place_id: crowding(occupancy[place_id], int(places[place_id]["capacity"])) for place_id in PLACE_IDS}


def candidate_attributes(place_id: str, origin: str, occupancy: dict[str, int], matrix: dict[str, dict[str, float]]) -> dict[str, float]:
    place = place_index()[place_id]
    return {
        "crowding": crowding(occupancy[place_id], int(place["capacity"])),
        "visual_attractiveness": float(place["visual_attractiveness"]),
        "novelty": float(place["novelty"]),
        "distance": travel_distance(matrix, origin, place_id),
    }
