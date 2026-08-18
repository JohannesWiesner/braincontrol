#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FIXME: This whole script has to be reworked. The general idea is
to test with empirical neuroimaging data derived from nilearn.

@author: johannes.wiesner
"""

from nilearn.datasets import fetch_localizer_button_task
from nilearn.datasets import fetch_localizer_contrasts
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
# User settings
###############################################################################

n_rois = 200
contrast_names = ["left button press (auditory cue)","right button press (auditory cue)",
                  "left button press (visual cue)","right button press (visual cue)"]

###############################################################################
# Download state maps
###############################################################################

data = fetch_localizer_contrasts(n_subjects=1,contrasts=contrast_names)
state_imgs_list = data['cmaps']
state_imgs_4d = load_img(state_imgs_list)
state_attributes = contrast_names.copy()

# create multiindex for states
state_attributes_multi = pd.DataFrame({'state':contrast_names})
state_attributes_multi[["hemisphere", "modality"]] = (
    state_attributes_multi["state"].str.extract(
        r"^(left|right).*?\((auditory|visual) cue\)$"
    )
)
state_attributes_multi = pd.MultiIndex.from_frame(state_attributes_multi)


###############################################################################
# Download atlas image
###############################################################################

atlas = fetch_atlas_schaefer_2018(
    n_rois=n_rois,
    yeo_networks=7,  # ENIGMA matrix uses the 7-network ordering
    resolution_mm=1,
)

atlas_img = atlas['maps']
atlas_lut = atlas['lut']

# create multiindex for nodes that can be later tested
atlas_labels_multi = atlas_lut.copy()
atlas_labels_multi[["hemisphere", "network"]] = atlas_labels_multi["name"].str.extract(
    r"^\d+Networks_(LH|RH)_(.+)_\d+$"
)
atlas_labels_multi = atlas_labels_multi.loc[:,['name','network','hemisphere']]
atlas_labels_multi = atlas_labels_multi[atlas_labels_multi['name'] != 'Background'].reset_index(drop=True)
atlas_labels_multi = pd.MultiIndex.from_frame(atlas_labels_multi)


###############################################################################
# download adjacency matrix
###############################################################################

# Pin the ENIGMA repository revision for reproducibility.
ENIGMA_REVISION = "b08974b55243060cbc1fad12c87048037446e8f7"

BASE_URL = (
    "https://raw.githubusercontent.com/MICA-MNI/ENIGMA/"
    f"{ENIGMA_REVISION}/enigmatoolbox/datasets/"
    "matrices/hcp_connectivity"
)

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

    return adjacency, labels

adjacency, matrix_labels = fetch_schaefer_structural_connectome(n_rois)

###############################################################################
# Sanity check that adjacency matrix and atlas are the same
###############################################################################

atlas_labels = np.asarray([
    label.decode() if hasattr(label, "decode") else str(label)
    for label in atlas.labels
])

atlas_labels = np.asarray([
    label
    for label in atlas_labels
    if label.strip().casefold() != "background"
])

if atlas_labels.shape != matrix_labels.shape:
    raise RuntimeError(
        f"Label counts differ: atlas={len(atlas_labels)}, "
        f"matrix={len(matrix_labels)}"
    )

if not np.array_equal(atlas_labels, matrix_labels):
    mismatch = np.flatnonzero(atlas_labels != matrix_labels)[0]
    raise RuntimeError(
        f"Label-order mismatch at node {mismatch}: "
        f"atlas={atlas_labels[mismatch]!r}, "
        f"matrix={matrix_labels[mismatch]!r}"
    )

###############################################################################
# Test braincontrol
###############################################################################

adjacency_df = pd.DataFrame(data=adjacency,index=atlas_labels,columns=atlas_labels)

# set masker
masker = NiftiLabelsMasker(labels_img=atlas_img,lut=atlas_lut,standardize=False)

###############################################################################
# Test when input is X
###############################################################################

transitioner_X = Transitioner(A=adjacency_df,T=1,masker=masker,
                              node_attributes=atlas_labels_multi,
                              state_attributes=state_attributes_multi,
                              order='permutations')

transitions_X_1 = transitioner_X.fit_transform(X=state_imgs_list)
transitions_X_2 = transitioner_X.fit_transform(X=state_imgs_4d)



transitioner_X = Transitioner(A=adjacency_df,T=1,masker=masker,
                              node_attributes=None,
                              state_attributes=None,
                              order='permutations')

X_df = pd.DataFrame(data=masker.fit_transform(state_imgs_list),
                    columns=atlas_labels_multi,
                    index=state_attributes_multi
                    )

transitions_X_3 = transitioner_X.fit_transform(X=X_df)

###############################################################################
# Test when input is x0 + xf
###############################################################################

state_attributes_x0xf_multi = state_attributes_multi[:2]

transitioner_x0xf = Transitioner(A=adjacency_df,T=1,masker=masker,
                              node_attributes=atlas_labels_multi,
                              state_attributes=state_attributes_x0xf_multi,
                              order='stability')

# FIXME: This has to be fixed, we want that the function still honors the order argument.
transitions_1 = transitioner_x0xf.fit_transform(x0=state_imgs_list[0],xf=state_imgs_list[1])

# # mask to get x0 and xf as arrays
x0_array = masker.fit_transform(state_imgs_list[0])
xf_array = masker.fit_transform(state_imgs_list[1])
transitions_4 = transitioner_x0xf.fit_transform(x0=x0_array,xf=xf_array)
