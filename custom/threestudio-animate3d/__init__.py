import threestudio
from packaging.version import Version

if hasattr(threestudio, "__version__") and Version(threestudio.__version__) >= Version(
    "0.2.1"
):
    pass
else:
    if hasattr(threestudio, "__version__"):
        print(f"[INFO] threestudio version: {threestudio.__version__}")
    raise ValueError(
        "threestudio version must be >= 0.2.0, please update threestudio by pulling the latest version from github"
    )

from . import data
from .guidance import animatemv_guidance
from .geometry import gaussian_4d, gaussian_3d_vis
from .renderer import diff_gaussian_rasterizer_advanced_4d
from .systems import animate3d