"""Copy the figures used by the documentation from data/ (ignored) to docs/figures/ (committed).

Run it after regenerating the data, then commit the updated images:
    python -m static_ads.tools.publish_figures
"""

import shutil

from ..paths import ANALYSIS_DIR, BASIN_DIR, DOCS_FIGURES, FIGURES_DIR, SCAN_DIR, STABILITY_DIR

FIGURES = {
    "basin.png": BASIN_DIR / "basin.png",  # Newton basins of the toy problem
    "stability_n1_k1.png": STABILITY_DIR / "stability_n1_k1.png",  # radial modes along n=1, k=1
    "surface_behaviour.png": FIGURES_DIR / "surface_behaviour.png",  # modes at the stellar surface
    "surface_behaviour.svg": FIGURES_DIR / "surface_behaviour.svg",
    "scan_overview.png": SCAN_DIR / "scan_overview.png",  # unstable modes vs rho_c, all families
    "scan_n5_k1.png": SCAN_DIR / "scan_n5_k1.png",  # the n = 5 spiral in detail
    "stability_all.png": ANALYSIS_DIR / "stability_all.png",  # stability of every stored run
}


def main():
    DOCS_FIGURES.mkdir(parents=True, exist_ok=True)
    for name, src in FIGURES.items():
        if src.exists():
            shutil.copy(src, DOCS_FIGURES / name)
            print(f"copied {name}")
        else:
            print(f"missing {src} (run the corresponding tool first)")


if __name__ == "__main__":
    main()
