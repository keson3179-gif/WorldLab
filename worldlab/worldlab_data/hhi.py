from __future__ import annotations


def herfindahl(shares: list[float]) -> float:
    return float(sum(share * share for share in shares))


def normalized_herfindahl(value: float, n_sites: int) -> float | None:
    """Normalized HHI on the unit interval. Undefined for a one-site group."""
    if n_sites <= 1:
        return None
    floor = 1.0 / n_sites
    return (value - floor) / (1.0 - floor)


def shares_from_visits(visits: list[float]) -> list[float] | None:
    """Shares of reported site counts. A missing member makes the month undefined."""
    if any(visit is None for visit in visits):
        return None
    total = float(sum(visits))
    if total == 0:
        return None
    return [float(visit) / total for visit in visits]
