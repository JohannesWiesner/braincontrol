"""Regressions for ordered state pairs through the public estimator API."""

import nibabel as nib
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from nctpy.energies import get_control_inputs, integrate_u
from nctpy.utils import matrix_normalization
from nilearn.maskers import NiftiLabelsMasker

from braincontrol.transitions import Transitioner


STATES = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.25]])
FINAL_STATES = np.array([[0.2, 0.8], [0.7, -0.1], [-0.3, 0.4]])


@pytest.fixture
def adjacency():
    """Return a small network with distinct node dynamics."""
    return np.array([[-1.0, 0.1], [0.1, -1.2]])


@pytest.fixture(params=[
    pytest.param(("directed", {"X": STATES},
                  [(0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)]),
                 id="directed"),
    pytest.param(("undirected", {"X": STATES}, [(0, 1), (0, 2), (1, 2)]),
                 id="undirected"),
    pytest.param(("directed_with_self", {"X": STATES},
                  [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2),
                   (2, 0), (2, 1), (2, 2)]), id="directed-with-self"),
    pytest.param(("self", {"X": STATES}, [(0, 0), (1, 1), (2, 2)]),
                 id="self"),
    pytest.param(("paired", {"X0": STATES[:2], "Xf": FINAL_STATES[:2]},
                  [(0, 0), (1, 1)]), id="paired"),
    pytest.param(("all_to_all", {"X0": STATES[:2], "Xf": FINAL_STATES[:2]},
                  [(0, 0), (0, 1), (1, 0), (1, 1)]), id="all-to-all-equal"),
    pytest.param(("all_to_all", {"X0": STATES[:2], "Xf": FINAL_STATES},
                  [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]),
                 id="all-to-all-more-final"),
    pytest.param(("all_to_all", {"X0": STATES, "Xf": FINAL_STATES[:2]},
                  [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)]),
                 id="all-to-all-more-initial"),
    pytest.param(("self", {"X": STATES[0]}, [(0, 0)]), id="self-vector"),
    pytest.param(("self", {"X": STATES[:1]}, [(0, 0)]), id="self-row"),
    pytest.param(("paired", {"X0": STATES[0], "Xf": FINAL_STATES[0]},
                  [(0, 0)]), id="paired-vectors"),
    pytest.param(("paired", {"X0": STATES[0], "Xf": FINAL_STATES[:1]},
                  [(0, 0)]), id="paired-vector-row"),
    pytest.param(("paired", {"X0": STATES[:1], "Xf": FINAL_STATES[0]},
                  [(0, 0)]), id="paired-row-vector"),
    pytest.param(("all_to_all", {"X0": STATES[0], "Xf": FINAL_STATES[0]},
                  [(0, 0)]), id="all-to-all-vectors"),
    pytest.param(("all_to_all", {"X0": STATES[0], "Xf": FINAL_STATES},
                  [(0, 0), (0, 1), (0, 2)]), id="all-to-all-single-initial"),
    pytest.param(("all_to_all", {"X0": STATES, "Xf": FINAL_STATES[0]},
                  [(0, 0), (1, 0), (2, 0)]), id="all-to-all-single-final"),
])
def transition_case(request):
    """Return states and explicit ordered pairs verified against dev #45."""
    return request.param


@pytest.mark.parametrize(("system", "T"), [("continuous", 0.01), ("discrete", 3)])
@pytest.mark.parametrize("energy_type", ["optimal", "minimal"])
def test_ordered_pairs_match_nctpy(adjacency, transition_case, system, T, energy_type):
    """Compare every pair's energy, trajectories and errors with the solver.

    The explicit pairs also pin the order of the pre-refactor dev implementation.
    Both estimator lifecycles must agree, including stored numerical results.
    """
    strategy, states, pairs = transition_case
    initial = np.atleast_2d(states.get("X", states.get("X0")))
    final = np.atleast_2d(states.get("X", states.get("Xf")))
    parameters = dict(
        T=T, system=system, transitions=strategy, energy_type=energy_type,
        rho=0.7 if energy_type == "optimal" else None,
        store_state_trajectories=True, store_control_trajectories=True,
    )
    B = np.diag([1.0, 0.8])
    S = np.diag([0.6, 0.9]) if energy_type == "optimal" else None
    reference = "midpoint" if energy_type == "optimal" else None
    inputs = dict(A=adjacency, B=B, S=S, xr=reference, **states)
    combined = Transitioner(**parameters)
    energy = combined.fit_transform(**inputs)
    separate = Transitioner(**parameters).fit(**inputs)
    np.testing.assert_allclose(energy, separate.transform(**inputs))
    xr.testing.assert_equal(combined.get_state_trajectories(), separate.get_state_trajectories())
    xr.testing.assert_equal(combined.get_control_trajectories(), separate.get_control_trajectories())
    np.testing.assert_array_equal(combined.get_errors(), separate.get_errors())

    trajectories = combined.get_state_trajectories().values
    controls = combined.get_control_trajectories().values
    assert energy.shape == (len(pairs), adjacency.shape[0])
    assert trajectories.shape == (round(T / 0.001) + 1 if system == "continuous" else T + 1,
                                  adjacency.shape[0], len(pairs))
    assert controls.shape == (trajectories.shape[0] if system == "continuous" else T,
                              adjacency.shape[0], len(pairs))
    for transition, (source, target) in enumerate(pairs):
        expected_x, expected_u, expected_errors = get_control_inputs(
            A_norm=matrix_normalization(adjacency, system=system, c=1),
            T=T, B=B, x0=initial[source], xf=final[target], system=system,
            rho=0.7 if energy_type == "optimal" else 1.0,
            S=S if S is not None else np.zeros_like(adjacency),
            xr=reference if reference is not None else "zero",
        )
        np.testing.assert_allclose(trajectories[:, :, transition], expected_x)
        np.testing.assert_allclose(controls[:, :, transition], expected_u)
        np.testing.assert_allclose(energy[transition], integrate_u(expected_u))
        np.testing.assert_allclose(combined.get_errors()[transition], expected_errors)


def test_pair_labels_follow_state_order(adjacency, transition_case):
    """Keep Series names, DataFrame rows and node labels in input order."""
    strategy, states, pairs = transition_case
    nodes = pd.Index(["right", "left"], name="node")
    labelled = {}
    labels = {}
    for name, values in states.items():
        labels[name] = [f"{name}-{i}" for i in range(np.atleast_2d(values).shape[0])]
        if values.ndim == 1:
            labelled[name] = pd.Series(values, index=nodes, name=labels[name][0])
        else:
            labelled[name] = pd.DataFrame(values, index=labels[name], columns=nodes)
    estimator = Transitioner(T=0.01, transitions=strategy, store_state_trajectories=True,
                             store_control_trajectories=True)
    energy = estimator.fit_transform(A=adjacency, **labelled)
    source_labels = labels.get("X", labels.get("X0"))
    target_labels = labels.get("X", labels.get("Xf"))
    expected_labels = [(source_labels[i], target_labels[j]) for i, j in pairs]
    assert energy.index.tolist() == expected_labels
    assert energy.columns.equals(nodes)
    for trajectory in (estimator.get_state_trajectories(), estimator.get_control_trajectories()):
        assert trajectory.coords["node"].values.tolist() == nodes.tolist()
        assert trajectory.coords["transition"].values.tolist() == expected_labels
    pd.testing.assert_frame_equal(energy, estimator.transform(A=adjacency, **labelled))
    expected = Transitioner(T=0.01, transitions=strategy).fit_transform(A=adjacency, **states)
    np.testing.assert_allclose(energy, expected)


def test_niimg_pairs_match_arrays(adjacency, transition_case):
    """Mask 3D/4D X, X0 and Xf inputs without changing the ordered results."""
    strategy, states, _ = transition_case
    labels_img = nib.Nifti1Image(
        np.array([1, 2], dtype=np.int16).reshape(2, 1, 1), np.eye(4)
    )
    masker = NiftiLabelsMasker(labels_img=labels_img, standardize=None, reports=False,
                             keep_masked_labels=False)
    images = {
        name: nib.Nifti1Image(
            values.reshape(2, 1, 1) if values.ndim == 1
            else values.T.reshape(2, 1, 1, values.shape[0]), np.eye(4)
        )
        for name, values in states.items()
    }
    parameters = dict(T=0.01, transitions=strategy, store_state_trajectories=True,
                      store_control_trajectories=True)
    image_estimator = Transitioner(**parameters, masker=masker)
    image_energy = image_estimator.fit_transform(A=adjacency, **images)
    array_estimator = Transitioner(**parameters)
    array_energy = array_estimator.fit_transform(A=adjacency, **states)
    np.testing.assert_allclose(image_energy, array_energy)
    xr.testing.assert_equal(image_estimator.get_state_trajectories(), array_estimator.get_state_trajectories())
    xr.testing.assert_equal(image_estimator.get_control_trajectories(), array_estimator.get_control_trajectories())
    separate = Transitioner(**parameters, masker=masker).fit(A=adjacency, **images)
    np.testing.assert_allclose(separate.transform(A=adjacency, **images), array_energy)
    assert not hasattr(masker, "n_elements_")


@pytest.mark.parametrize("method", ["fit", "transform"])
@pytest.mark.parametrize("single", [STATES[0], STATES[:1]])
@pytest.mark.parametrize("strategy", ["directed", "undirected", "directed_with_self", "paired", "all_to_all"])
def test_single_X_rejects_other_strategies(adjacency, method, single, strategy):
    """One state in X still requires self, regardless of public dimensions."""
    estimator = Transitioner(T=0.01, transitions=strategy)
    if method == "transform":
        valid = {"X0": STATES, "Xf": FINAL_STATES} if strategy in ("paired", "all_to_all") else {"X": STATES}
        estimator.fit(A=adjacency, **valid)
    with pytest.raises(ValueError, match="not compatible with 1 state"):
        getattr(estimator, method)(A=adjacency, X=single)


@pytest.mark.parametrize("method", ["fit", "transform"])
@pytest.mark.parametrize(("strategy", "states", "message"), [
    ("paired", {"X0": STATES, "Xf": FINAL_STATES[:2]}, "same number of states"),
    ("paired", {"X0": STATES[0], "Xf": FINAL_STATES}, "same number of states"),
    ("paired", {"X0": STATES, "Xf": FINAL_STATES[0]}, "same number of states"),
    ("paired", {"X": STATES}, "provided through X"),
    ("all_to_all", {"X": STATES}, "provided through X"),
    ("self", {"X0": STATES, "Xf": FINAL_STATES}, "provided through X0 and Xf"),
    ("directed", {"X0": STATES, "Xf": FINAL_STATES}, "provided through X0 and Xf"),
    ("undirected", {"X0": STATES, "Xf": FINAL_STATES}, "provided through X0 and Xf"),
    ("directed_with_self", {"X0": STATES, "Xf": FINAL_STATES}, "provided through X0 and Xf"),
    ("paired", {}, "Provide either X or both X0 and Xf"),
    ("paired", {"X0": STATES}, "must be provided together"),
    ("paired", {"Xf": STATES}, "must be provided together"),
    ("paired", {"X": STATES, "X0": STATES, "Xf": FINAL_STATES}, "not both"),
])
def test_invalid_state_combinations(adjacency, method, strategy, states, message):
    """Preserve strategy and input validation during both estimator stages."""
    estimator = Transitioner(T=0.01, transitions=strategy)
    if method == "transform":
        valid = {"X0": STATES, "Xf": FINAL_STATES} if strategy in ("paired", "all_to_all") else {"X": STATES}
        estimator.fit(A=adjacency, **valid)
    with pytest.raises(ValueError, match=message):
        getattr(estimator, method)(A=adjacency, **states)


@pytest.mark.parametrize("strategy", ["paired", "all_to_all"])
def test_separate_multiindex_labels(adjacency, strategy):
    """Preserve each hierarchical level for distinct initial/final labels."""
    nodes = pd.MultiIndex.from_tuples([("R", "a"), ("L", "b")], names=["hemisphere", "region"])
    initial_labels = pd.MultiIndex.from_tuples([("pre", "task"), ("pre", "rest")], names=["visit", "state"])
    final_labels = pd.MultiIndex.from_tuples([("post", "rest"), ("post", "task"), ("post", "recovery")], names=["visit", "state"])
    if strategy == "paired":
        final_labels = final_labels[:2]
    X0 = pd.DataFrame(STATES[:2], index=initial_labels, columns=nodes)
    Xf = pd.DataFrame(FINAL_STATES[:len(final_labels)], index=final_labels, columns=nodes)
    estimator = Transitioner(T=0.01, transitions=strategy, store_state_trajectories=True)
    energy = estimator.fit_transform(A=adjacency, X0=X0, Xf=Xf)
    pairs = [(0, 0), (1, 1)] if strategy == "paired" else [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]
    expected = pd.MultiIndex.from_tuples([
        tuple((initial_labels[i][level], final_labels[j][level]) for level in range(2))
        for i, j in pairs
    ], names=["visit", "state"])
    pd.testing.assert_index_equal(energy.index, expected)
    pd.testing.assert_index_equal(energy.columns, nodes)
    pd.testing.assert_index_equal(estimator.get_state_trajectories().indexes["transition"], expected)
    pd.testing.assert_index_equal(estimator.get_state_trajectories().indexes["node"], nodes)


@pytest.mark.parametrize("method", ["fit", "transform"])
def test_separate_states_reject_reordered_nodes(adjacency, method):
    """Node validation must compare each original state collection's labels."""
    X0 = pd.DataFrame(STATES, columns=["right", "left"])
    Xf = pd.DataFrame(FINAL_STATES, columns=["right", "left"])
    estimator = Transitioner(T=0.01, transitions="paired")
    if method == "transform":
        estimator.fit(A=adjacency, X0=X0, Xf=Xf)
    with pytest.raises(ValueError, match="Node labels must match across all inputs"):
        getattr(estimator, method)(A=adjacency, X0=X0, Xf=Xf.iloc[:, ::-1])


def test_transform_checks_schema_and_uses_new_state_sets(adjacency):
    """Fitting records node schema without retaining a split or state count."""
    estimator = Transitioner(T=0.01, transitions="all_to_all").fit(
        A=adjacency, X0=STATES[0], Xf=FINAL_STATES[0]
    )
    actual = estimator.transform(A=adjacency, X0=STATES[:2], Xf=FINAL_STATES)
    expected = Transitioner(T=0.01, transitions="all_to_all").fit_transform(
        A=adjacency, X0=STATES[:2], Xf=FINAL_STATES
    )
    np.testing.assert_allclose(actual, expected)
    with pytest.raises(ValueError, match="fitted schema contains 2 nodes"):
        estimator.transform(A=np.eye(3), X0=np.ones((2, 3)), Xf=np.ones((3, 3)))
