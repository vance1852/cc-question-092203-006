"""约束检查模块：租赁边界、禁建区可行域与间距约束。"""

from .boundary import (
    SiteBoundary,
    GeometryValidationError,
    create_rectangular_boundary,
    create_hexagonal_boundary,
    create_irregular_boundary,
)
from .exclusion import (
    ExclusionZone,
    FeasibleDomain,
    LayoutViolationReport,
    InfeasibleLayoutError,
    ZONE_KINDS,
)
from .spacing import (
    check_min_spacing,
    compute_min_spacing_from_diameters,
    compute_pairwise_distances,
    enforce_min_spacing,
)

__all__ = [
    "SiteBoundary",
    "GeometryValidationError",
    "create_rectangular_boundary",
    "create_hexagonal_boundary",
    "create_irregular_boundary",
    "ExclusionZone",
    "FeasibleDomain",
    "LayoutViolationReport",
    "InfeasibleLayoutError",
    "ZONE_KINDS",
    "check_min_spacing",
    "compute_min_spacing_from_diameters",
    "compute_pairwise_distances",
    "enforce_min_spacing",
]
