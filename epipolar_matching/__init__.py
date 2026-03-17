"""
epipolar_matching
=================
Pixel-accurate epipolar guided matching via angular segment tree.

Build the C++ extension first:
    Windows : build.bat
    Linux   : bash build.sh
    pip     : pip install -e .

Then import:
    from epipolar_matching import (
        epipolar_wedge_filter_with_segment_tree,
        epipolar_geometric_distance_filter,
    )
"""

try:
    from .guided_desc_match import (
        epipolar_wedge_filter_with_segment_tree,
        epipolar_wedge_filter_with_segment_tree_origin,
        epipolar_geometric_distance_filter,
        epipolar_hash_filter_cpp,
    )
except ImportError as _e:
    raise ImportError(
        "guided_desc_match C++ extension not found.\n"
        "Build it first:\n"
        "  Windows : build.bat\n"
        "  Linux   : bash build.sh\n"
        "  pip     : pip install -e .\n\n"
        f"Original error: {_e}"
    ) from _e

__all__ = [
    "epipolar_wedge_filter_with_segment_tree",
    "epipolar_wedge_filter_with_segment_tree_origin",
    "epipolar_geometric_distance_filter",
    "epipolar_hash_filter_cpp",
]
