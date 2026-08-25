#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utilities to get example datasets for braincontrol

@author: johannes.wiesner
"""

from nilearn.datasets import fetch_localizer_button_task
from nilearn.datasets import fetch_localizer_contrasts
from nilearn.datasets import fetch_neurovault_ids



from braincontrol.transitions import Transitioner
from nilearn.maskers import NiftiLabelsMasker
from nilearn.plotting import plot_stat_map
from nilearn.datasets import fetch_atlas_schaefer_2018
from pathlib import Path
from shutil import copyfileobj
from urllib.request import Request, urlopen
import numpy as np
import pandas as pd
from nilearn.image import load_img

###############################################################################
# Constants
###############################################################################

# for adjacency matrix
ENIGMA_REVISION = "b08974b55243060cbc1fad12c87048037446e8f7"
BASE_URL = (
    "https://raw.githubusercontent.com/MICA-MNI/ENIGMA/"
    f"{ENIGMA_REVISION}/enigmatoolbox/datasets/"
    "matrices/hcp_connectivity"
)

# for statistical maps
# TODO: Needs to be reworked, too many duplicates. If there are multiple 
# stat maps with the same contrast use the one with the larger number_of_subjects attribute
# TODO: All FOVs must be the same and all images must be in MNI space!
NEUROVAULT_DF = pd.DataFrame([
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
    #[3128, "Negative Valence Systems", 'Acute Threat ("Fear")', None, "Faces minus Shapes"],
    #[3135, "Negative Valence Systems", "Loss", None, "Punish"],
    # [3180, "Social Processes", "Perception and Understanding of Others", "Understanding Mental States", "Mental Interaction minus Random Interaction"],
    #[3181, "Social Processes", "Perception and Understanding of Others", "Understanding Mental States", "Mental Interaction"],
    #[805923, "Social Processes", "Affiliation and Attachment", None, "cooperation > defection"],
    #[805925, "Social Processes", "Affiliation and Attachment", None, "cooperation > defection"],
    #[805924, "Social Processes", "Affiliation and Attachment", None, "defection  >  cooperation"],
    #[3157, "Sensorimotor Systems", "Motor Actions", "Execution", "Left Finger (Hand) minus Average"],
    #[3161, "Sensorimotor Systems", "Motor Actions", "Execution", "Right Finger (Hand) minus Average"],
    #[3155, "Sensorimotor Systems", "Motor Actions", "Execution", "Left Toe (Foot) minus Average"],
    #[550241, "Sensorimotor Systems", "Motor Actions", "Execution", "Go trials (sum of left hand responses and right hand responses) versus baseline at the time of responses."],
], columns=["id", "rdoc_domain", "rdoc_construct", "rdoc_subconstruct", "contrast_definition"])

###############################################################################
# Get structural adjacency matrix
###############################################################################

def fetch_schaefer_structural_connectome(
    n_rois: int = 200,
    cache_dir: str | Path = ".cache/enigma",
) -> tuple[np.ndarray, np.ndarray]:
    """Download an ENIGMA/HCP Schaefer structural adjacency matrix."""

    if n_rois not in {100, 200, 300, 400}:
        raise ValueError("n_rois must be 100, 200, 300, or 400")

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    matrix_name = f"strucMatrix_ctx_schaefer_{n_rois}.csv"
    labels_name = f"strucLabels_ctx_schaefer_{n_rois}.csv"

    def download(filename: str) -> Path:
        destination = cache_dir / filename

        if not destination.exists():
            request = Request(
                f"{BASE_URL}/{filename}",
                headers={"User-Agent": "Python structural-connectome downloader"},
            )
            temporary = destination.with_suffix(destination.suffix + ".part")

            with urlopen(request, timeout=60) as response:
                with temporary.open("wb") as output:
                    copyfileobj(response, output)

            temporary.replace(destination)

        return destination

    matrix_file = download(matrix_name)
    labels_file = download(labels_name)

    adjacency = np.loadtxt(matrix_file, delimiter=",", dtype=float)
    labels = np.loadtxt(
        labels_file,
        delimiter=",",
        dtype=str,
        ndmin=1,
    )

    if adjacency.shape != (n_rois, n_rois):
        raise RuntimeError(f"Unexpected matrix shape: {adjacency.shape}")

    if labels.shape != (n_rois,):
        raise RuntimeError(f"Unexpected label shape: {labels.shape}")
    
    adjacency = pd.DataFrame(
    adjacency,
    columns=labels,
    index=labels,
    ).rename_axis(index="name", columns="name")

    return adjacency

###############################################################################
# Download state maps
###############################################################################

def fetch_neurovault_stat_maps(image_ids=None):
    """Download NeuroVault statistical maps and return their metadata and paths.

    If image_ids is provided, fetch those images and return their IDs, paths,
    and contrast definitions. Otherwise, fetch images from NEUROVAULT_DF and
    append their downloaded paths to the existing metadata.
    """
    if image_ids is not None:
        data = fetch_neurovault_ids(image_ids=list(image_ids))

        return pd.DataFrame({
            "id": [meta["id"] for meta in data.images_meta],
            "path": data.images,
            "contrast_definition": [
                meta.get("contrast_definition") for meta in data.images_meta
            ],
        })

    else:
        data = fetch_neurovault_ids(image_ids=NEUROVAULT_DF["id"].tolist())

        paths = pd.DataFrame({
            "id": [meta["id"] for meta in data.images_meta],
            "path": data.images,
        })

        return NEUROVAULT_DF.merge(paths, on="id", how="left")