"""Opt-in integration tests using empirical neuroimaging data.

Set ``BRAINCONTROL_RUN_NEUROIMAGING_TESTS=1`` to enable these tests. They
download Nilearn localizer data, a Schaefer atlas, and a pinned ENIGMA
structural connectome, so they are skipped during the normal unit-test run.
"""

import os
from shutil import copyfileobj
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
import pytest
from nilearn.datasets import fetch_atlas_schaefer_2018
from nilearn.datasets import fetch_localizer_contrasts
from nilearn.image import load_img
from nilearn.maskers import NiftiLabelsMasker

from braincontrol.transitions import Transitioner


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("BRAINCONTROL_RUN_NEUROIMAGING_TESTS") != "1",
        reason=(
            "set BRAINCONTROL_RUN_NEUROIMAGING_TESTS=1 to run networked "
            "neuroimaging integration tests"
        ),
    ),
]

N_ROIS = 100
CONTRAST_NAMES = [
    "left button press (auditory cue)",
    "right button press (auditory cue)",
    "left button press (visual cue)",
    "right button press (visual cue)",
]
ENIGMA_REVISION = "b08974b55243060cbc1fad12c87048037446e8f7"
ENIGMA_BASE_URL = (
    "https://raw.githubusercontent.com/MICA-MNI/ENIGMA/"
    f"{ENIGMA_REVISION}/enigmatoolbox/datasets/"
    "matrices/hcp_connectivity"
)


def _download(url, destination):
    """Download one integration-test resource into its cache."""
    if destination.exists():
        return destination

    request = Request(url, headers={"User-Agent": "braincontrol-tests"})
    temporary = destination.with_suffix(destination.suffix + ".part")
    with urlopen(request, timeout=60) as response:
        with temporary.open("wb") as output:
            copyfileobj(response, output)
    temporary.replace(destination)
    return destination


def _fetch_schaefer_structural_connectome(n_rois, cache_dir):
    """Fetch a pinned ENIGMA/HCP Schaefer structural connectome."""
    if n_rois not in {100, 200, 300, 400}:
        raise ValueError("n_rois must be 100, 200, 300, or 400")

    cache_dir.mkdir(parents=True, exist_ok=True)
    matrix_name = f"strucMatrix_ctx_schaefer_{n_rois}.csv"
    labels_name = f"strucLabels_ctx_schaefer_{n_rois}.csv"
    matrix_file = _download(
        f"{ENIGMA_BASE_URL}/{matrix_name}", cache_dir / matrix_name
    )
    labels_file = _download(
        f"{ENIGMA_BASE_URL}/{labels_name}", cache_dir / labels_name
    )

    adjacency = np.loadtxt(matrix_file, delimiter=",", dtype=float)
    labels = np.loadtxt(labels_file, delimiter=",", dtype=str, ndmin=1)
    if adjacency.shape != (n_rois, n_rois):
        raise RuntimeError(f"Unexpected matrix shape: {adjacency.shape}")
    if labels.shape != (n_rois,):
        raise RuntimeError(f"Unexpected label shape: {labels.shape}")
    return adjacency, labels


def _state_labels():
    """Return hierarchical labels for the empirical contrast maps."""
    frame = pd.DataFrame({"state": CONTRAST_NAMES})
    frame[["hemisphere", "modality"]] = frame["state"].str.extract(
        r"^(left|right).*?\((auditory|visual) cue\)$"
    )
    return pd.MultiIndex.from_frame(frame)


def _node_labels(atlas_lut):
    """Return hierarchical labels for non-background atlas regions."""
    frame = atlas_lut.copy()
    frame[["hemisphere", "network"]] = frame["name"].str.extract(
        r"^\d+Networks_(LH|RH)_(.+)_\d+$"
    )
    frame = frame.loc[frame["name"] != "Background"]
    return pd.MultiIndex.from_frame(
        frame.loc[:, ["name", "network", "hemisphere"]].reset_index(drop=True)
    )


@pytest.fixture(scope="module")
def empirical_data(tmp_path_factory):
    """Download and prepare all empirical integration-test inputs."""
    cache_dir = tmp_path_factory.mktemp("braincontrol-neuroimaging")
    localizer = fetch_localizer_contrasts(
        n_subjects=1,
        contrasts=CONTRAST_NAMES,
        data_dir=cache_dir / "nilearn",
    )
    state_images = localizer["cmaps"]
    atlas = fetch_atlas_schaefer_2018(
        n_rois=N_ROIS,
        yeo_networks=7,
        resolution_mm=1,
        data_dir=cache_dir / "nilearn",
    )
    adjacency, matrix_labels = _fetch_schaefer_structural_connectome(
        N_ROIS, cache_dir / "enigma"
    )

    atlas_labels = [
        label.decode() if hasattr(label, "decode") else str(label)
        for label in atlas.labels
    ]
    atlas_labels = np.asarray(
        [
            label
            for label in atlas_labels
            if label.strip().casefold() != "background"
        ]
    )
    np.testing.assert_array_equal(atlas_labels, matrix_labels)

    labels = _node_labels(atlas["lut"])
    masker = NiftiLabelsMasker(
        labels_img=atlas["maps"],
        lut=atlas["lut"],
        standardize=False,
        reports=False,
        keep_masked_labels=False,
    )
    return {
        "adjacency": pd.DataFrame(
            adjacency, index=matrix_labels, columns=matrix_labels
        ),
        "images": state_images,
        "image_4d": load_img(state_images),
        "masker": masker,
        "node_labels": labels,
        "state_labels": _state_labels(),
    }


def _transitioner(empirical_data, **kwargs):
    """Construct an integration-test transformer."""
    parameters = {
        "A": empirical_data["adjacency"],
        "T": 0.002,
        "masker": empirical_data["masker"],
        "node_labels": empirical_data["node_labels"],
        "state_labels": empirical_data["state_labels"],
        "order": "permutations",
        "store_trajectories": False,
        "store_control_trajectories": False,
    }
    parameters.update(kwargs)
    return Transitioner(**parameters)


def _transition_label(source, target):
    """Return one endpoint-paired hierarchical transition label."""
    return (
        ("source", "target"),
        *zip(source, target),
    )


def test_empirical_image_and_dataframe_inputs_match(empirical_data):
    """List, 4D-image, and pre-masked inputs should produce equal energies."""
    from_list = _transitioner(empirical_data).fit_transform(
        X=empirical_data["images"]
    )
    from_4d = _transitioner(empirical_data).fit_transform(
        X=empirical_data["image_4d"]
    )
    masked = empirical_data["masker"].fit_transform(empirical_data["images"])
    masked_df = pd.DataFrame(
        masked,
        index=empirical_data["state_labels"],
        columns=empirical_data["node_labels"],
    )
    from_dataframe = Transitioner(
        A=empirical_data["adjacency"],
        T=0.002,
        order="permutations",
        store_trajectories=False,
        store_control_trajectories=False,
    ).fit_transform(X=masked_df)

    pd.testing.assert_frame_equal(from_list, from_4d)
    pd.testing.assert_frame_equal(from_list, from_dataframe)


def test_empirical_endpoint_inputs_honor_order(empirical_data):
    """Product order should compute both directions and both self transitions."""
    labels = empirical_data["state_labels"][:2]
    transitioner = _transitioner(
        empirical_data,
        order="product",
        state_labels=labels,
    )

    result = transitioner.fit_transform(
        x0=empirical_data["images"][0],
        xf=empirical_data["images"][1],
    )

    assert transitioner.transition_indices_ == [
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
    ]
    expected_index = pd.MultiIndex.from_tuples(
        [
            _transition_label(labels[0], labels[0]),
            _transition_label(labels[0], labels[1]),
            _transition_label(labels[1], labels[0]),
            _transition_label(labels[1], labels[1]),
        ],
        names=result.index.names,
    )
    pd.testing.assert_index_equal(result.index, expected_index)
