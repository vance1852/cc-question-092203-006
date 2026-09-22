"""约束检查模块：租赁边界、禁建区可行域与风机间距。"""

from .boundary import (
    SiteBoundary,
    validate_polygon_geometry,
    signed_area,
    create_rectangular_boundary,
    create_hexagonal_boundary,
    create_irregular_boundary,
)
from .geofence import (
    Geofence,
    ExclusionZone,
    ConstraintViolation,
    InfeasibleLayoutError,
)
from .spacing import (
    check_min_spacing,
    compute_min_spacing_from_diameters,
    compute_pairwise_distances,
    enforce_min_spacing,
)

__all__ = [
    "SiteBoundary",
    "validate_polygon_geometry",
    "signed_area",
    "create_rectangular_boundary",
    "create_hexagonal_boundary",
    "create_irregular_boundary",
    "Geofence",
    "ExclusionZone",
    "ConstraintViolation",
    "InfeasibleLayoutError",
    "check_min_spacing",
    "compute_min_spacing_from_diameters",
    "compute_pairwise_distances",
    "enforce_min_spacing",
]
