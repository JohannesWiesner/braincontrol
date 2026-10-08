
"""Utilities for fetching example datasets for braincontrol."""

from pathlib import Path
from shutil import copyfileobj
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
from nilearn.datasets import fetch_neurovault_ids

###############################################################################
## Constants
###############################################################################

# constants for downloading structural connectome
ENIGMA_REVISION = "b08974b55243060cbc1fad12c87048037446e8f7"

BASE_URL = (
    "https://raw.githubusercontent.com/MICA-MNI/ENIGMA/"
    f"{ENIGMA_REVISION}/enigmatoolbox/datasets/"
    "matrices/hcp_connectivity"
)

# TODO: Remove duplicate contrasts, preferring the image with the
# largest number_of_subjects when multiple maps describe the same contrast.
# TODO: Verify that all statistical maps share a field of view and
# are in MNI space.
# constants for download state maps
NEUROVAULT_DF = pd.DataFrame(
    [
        [3190, "Cognitive Systems", "Working Memory", "Flexible Updating", "2-back minus 0-back"],
        [8820, "Cognitive Systems", "Cognitive Control", "Goal Selection; Updating, Representation, and Maintenance", "Relational Processing minus Matching"],
        [3142, "Cognitive Systems", "Language", None, "Story minus Math"],
        [151, "Cognitive Systems", "Cognitive Control", "Response Selection; Inhibition/Suppression", "successful stop minus go"],
        [3041, "Cognitive Systems", "Cognitive Control", "Response Selection; Inhibition/Suppression", "succstop minus go"],
        [3042, "Cognitive Systems", "Cognitive Control", "Response Selection; Inhibition/Suppression", "succ stop minus go"],
        [3136, "Positive Valence Systems", "Reward Responsiveness", "Initial Response to Reward", "Reward minus Punish"],
        [3137, "Positive Valence Systems", "Reward Responsiveness", "Initial Response to Reward", "Reward"],
        [550248, "Positive Valence Systems", "Reward Learning", "Reward Prediction Error", "Standard reward prediction errors (parametric modulation)"],
        [550249, "Positive Valence Systems", "Reward Learning", "Reward Prediction Error", "Biased minus standard reward prediction errors (parametric modulation)"],
        [550239, "Positive Valence Systems", "Reward Responsiveness", "Initial Response to Reward", "Correlation with reward outcomes (+1), neutral outcomes (0), punishment outcomes (-1)."],
    ],
    columns=[
        "id",
        "rdoc_domain",
        "rdoc_construct",
        "rdoc_subconstruct",
        "contrast_definition",
    ],
)

###############################################################################
## Structural connectome
###############################################################################

def _download_file(url, destination):
    """Download a file unless it already exists in the cache.

    Files are first written to a temporary path to avoid leaving
    incomplete downloads at the expected destination.
    """
    if destination.exists():
        return destination

    temporary = destination.with_suffix(
        destination.suffix + ".part"
    )

    request = Request(
        url,
        headers={"User-Agent": "braincontrol"},
    )

    try:
        with urlopen(request, timeout=60) as response:
            with temporary.open("wb") as output:
                copyfileobj(response, output)

        temporary.replace(destination)

    finally:
        temporary.unlink(missing_ok=True)

    return destination

# TODO: Would be nice if nilearn would offer this, so we don't need to 
# download from a Github repo. I opened an issue:
# https://github.com/nilearn/nilearn/issues/6523
def fetch_schaefer_structural_connectome(
    n_rois: int = 200,
    cache_dir: str | Path = ".cache/enigma",
) -> pd.DataFrame:
    """Fetch a Schaefer structural connectome from ENIGMA/HCP.

    The adjacency matrix is returned as a DataFrame with Schaefer
    region labels on both axes. Downloaded files are cached locally.
    """
    if n_rois not in {100, 200, 300, 400}:
        raise ValueError(
            "n_rois must be 100, 200, 300, or 400"
        )

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    matrix_name = (
        f"strucMatrix_ctx_schaefer_{n_rois}.csv"
    )
    labels_name = (
        f"strucLabels_ctx_schaefer_{n_rois}.csv"
    )

    matrix_file = _download_file(
        f"{BASE_URL}/{matrix_name}",
        cache_dir / matrix_name,
    )

    labels_file = _download_file(
        f"{BASE_URL}/{labels_name}",
        cache_dir / labels_name,
    )

    adjacency = np.loadtxt(
        matrix_file,
        delimiter=",",
        dtype=float,
    )

    labels = np.loadtxt(
        labels_file,
        delimiter=",",
        dtype=str,
        ndmin=1,
    )

    if adjacency.shape != (n_rois, n_rois):
        raise RuntimeError(
            f"Unexpected matrix shape: {adjacency.shape}"
        )

    if labels.shape != (n_rois,):
        raise RuntimeError(
            f"Unexpected label shape: {labels.shape}"
        )

    return pd.DataFrame(
        adjacency,
        index=pd.Index(labels, name="name"),
        columns=pd.Index(labels, name="name"),
    )


###############################################################################
## Statistical maps
###############################################################################

def fetch_neurovault_stat_maps(image_ids=None):
    """Fetch NeuroVault statistical maps and their metadata.

    When ``image_ids`` is omitted, fetch the predefined maps and
    return their RDoC metadata. Otherwise, fetch the requested maps
    and return their IDs, paths, and contrast definitions.
    """
    if image_ids is None:
        image_ids = NEUROVAULT_DF["id"].tolist()
        predefined = True
    else:
        image_ids = list(image_ids)
        predefined = False

    data = fetch_neurovault_ids(
        image_ids=image_ids,
    )

    metadata = pd.DataFrame(
        {
            "id": [
                meta["id"]
                for meta in data.images_meta
            ],
            "path": data.images,
            "contrast_definition": [
                meta.get("contrast_definition")
                for meta in data.images_meta
            ],
        }
    )

    if predefined:
        return NEUROVAULT_DF.merge(
            metadata[["id", "path"]],
            on="id",
            how="left",
        )

    return metadata