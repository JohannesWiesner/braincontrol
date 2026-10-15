import nibabel as nib
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from nctpy.energies import get_control_inputs, integrate_u
from nilearn._utils.cache_mixin import CacheMixin
from nilearn.maskers import NiftiLabelsMasker
from sklearn.exceptions import NotFittedError

import braincontrol.transitions as transitions
from braincontrol.transitions import Transitioner
from braincontrol.utils.processing.transitions import (
    _get_trajectory_array,
    _get_transition_energy,
    _get_transition_trajectories,
)


####################
## Fixtures
####################

@pytest.fixture
def transition_data():
    """Return a small unlabelled network and three state vectors.
    
    The fixture provides deterministic NumPy inputs that are reused across tests
    of transition generation, estimator fitting, and numerical output."""
    A = np.array([[-1.0, 0.1], [0.1, -1.2]])
    X = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.25]])
    return A, X

@pytest.fixture
def labelled_transition_data(transition_data):
    """Return the standard test data with pandas node and state labels.
    
    The labelled representation is used to verify that node schema and state
    metadata are preserved correctly through fitting and transformation."""
    A, X = transition_data
    nodes = pd.Index(["left", "right"], name="node")
    states = pd.Index(["rest", "task", "recovery"], name="state")
    A = pd.DataFrame(A, index=nodes, columns=nodes)
    X = pd.DataFrame(X, index=states, columns=nodes)
    return A, X

####################
## Transition processing
####################

def test_transition_trajectories_match_nctpy(transition_data):
    """Test trajectory computation against a direct ``nctpy`` calculation.
    
    The helper should return the same state trajectory, control trajectory, and
    numerical error as ``nctpy.get_control_inputs`` for an equivalent transition."""
    A, X = transition_data
    xs, us, errors = _get_transition_trajectories(
        A, X[:2], X[1::-1], 0.002, np.eye(2), 1.0, np.eye(2)
    )
    expected_x, expected_u, expected_error = get_control_inputs(
        A_norm=A, T=0.002, B=np.eye(2), x0=X[0], xf=X[1],
        system="continuous", rho=1.0, S=np.eye(2), xr="xf",
    )
    assert len(xs) == 2
    assert len(us) == 2
    assert errors.shape == (2, 2)
    np.testing.assert_allclose(xs[0], expected_x)
    np.testing.assert_allclose(us[0], expected_u)
    np.testing.assert_allclose(errors[0], expected_error)


def test_transition_energy_matches_nctpy(transition_data):
    """Test node-level energy integration against ``nctpy``.
    
    The transition-energy helper must integrate each control trajectory in the
    same way as ``nctpy.integrate_u`` and return one energy value per node."""
    A, X = transition_data
    _, controls, _ = _get_transition_trajectories(
        A, X[:2], X[1::-1], 0.002, np.eye(2), 1.0, np.eye(2)
    )
    energy = _get_transition_energy(controls)
    assert energy.shape == (2, 2)
    np.testing.assert_allclose(energy[0], integrate_u(controls[0]))

def test_get_trajectory_array_handles_named_index():
    """Test conversion of stored trajectories to a labelled xarray object.
    
    Named pandas indices, including tuple-valued transition labels, must become
    coordinates without changing the expected time-node-transition dimensions."""
    transition_values = np.empty(
        2,
        dtype=object,
    )
    transition_values[:] = [
        ("rest", "task"),
        ("task", "rest"),
    ]

    transition_labels = pd.Index(
        transition_values,
        name="transition",
    )

    result = _get_trajectory_array(
        [
            np.ones((3, 2)),
            np.zeros((3, 2)),
        ],
        node_labels=pd.Index(
            ["A", "B"],
            name="name",
        ),
        transition_labels=transition_labels,
        name="trajectories",
    )

    assert isinstance(result, xr.DataArray)
    assert result.dims == (
        "time",
        "node",
        "transition",
    )
    assert result.shape == (3, 2, 2)
    assert result.name == "trajectories"
    assert result.coords["node"].values.tolist() == [
        "A",
        "B",
    ]
    assert result.coords["transition"].values.tolist() == [
        ("rest", "task"),
        ("task", "rest"),
    ]


####################
## Estimator API and fitting
####################

def test_transitioner_public_api():
    """Test the minimal public estimator API.
    
    ``Transitioner`` should be the exported public object from this module and
    retain Nilearn's cache mixin required by its estimator implementation."""
    assert transitions.__all__ == ["Transitioner"]
    assert issubclass(Transitioner, CacheMixin)


@pytest.mark.parametrize(
    ("parameter", "value", "error"),
    [
        ("energy_type", "invalid", ValueError),
        ("system", "invalid", ValueError),
        ("expm_version", "invalid", ValueError),
        ("transitions", "invalid", ValueError),
        ("normalize_A", "yes", TypeError),
        ("c", 0, ValueError),
        ("store_state_trajectories", 1, TypeError),
        ("store_control_trajectories", "yes", TypeError),
    ],
)
def test_constructor_parameter_validation(transition_data, parameter, value, error):
    """Test validation of invalid constructor parameters during fitting.
    
    Each parameter/value pair represents an invalid estimator configuration and
    must raise the corresponding exception before an NCT calculation is run."""
    A, X = transition_data
    estimator = Transitioner(T=0.002, **{parameter: value})
    with pytest.raises(error):
        estimator.fit(A=A, X=X)


def test_fit_stores_node_schema(transition_data):
    """Test the fitted schema learned from unlabelled NumPy inputs.
    
    Fitting should record the common node count and transition strategy while
    leaving ``node_labels_`` unset when no labels are available."""
    A, X = transition_data
    estimator = Transitioner(T=0.002).fit(A=A, X=X)
    assert estimator.n_nodes_ == 2
    assert estimator.node_labels_ is None
    assert estimator.transitions_ == "directed"


def test_fit_stores_labelled_node_schema(labelled_transition_data):
    """Test the fitted schema learned from labelled pandas inputs.
    
    The estimator should retain the common node labels established during
    fitting so that later transform inputs can be checked against them."""
    A, X = labelled_transition_data
    estimator = Transitioner(T=0.002).fit(A=A, X=X)
    assert estimator.n_nodes_ == 2
    assert estimator.node_labels_.equals(X.columns)


def test_fit_transform_matches_fit_then_transform(transition_data):
    """Test the custom ``fit_transform`` estimator lifecycle.
    
    Calling ``fit_transform`` directly must produce the same control energies as
    calling ``fit`` followed by ``transform`` with the same empirical inputs."""
    A, X = transition_data
    combined = Transitioner(T=0.002, transitions="undirected").fit_transform(A=A, X=X)
    estimator = Transitioner(T=0.002, transitions="undirected").fit(A=A, X=X)
    separate = estimator.transform(A=A, X=X)
    np.testing.assert_allclose(combined, separate)


####################
## Reference states
####################

@pytest.mark.parametrize(
    "reference",
    [
        pytest.param(np.array([0.2, 0.4]), id="array-1d"),
        pytest.param(np.array([[0.2, 0.4]]), id="array-2d"),
        pytest.param([0.2, 0.4], id="list-1d"),
        pytest.param([[0.2, 0.4]], id="list-2d"),
        pytest.param((0.2, 0.4), id="tuple-1d"),
        pytest.param(((0.2, 0.4),), id="tuple-2d"),
        pytest.param(pd.Series([0.2, 0.4], index=["left", "right"]), id="series"),
        pytest.param(pd.DataFrame([[0.2, 0.4]], columns=["left", "right"]), id="dataframe"),
    ],
)
@pytest.mark.parametrize(("system", "T"), [("continuous", 0.01), ("discrete", 3)])
def test_xr_array_like_matches_nctpy(labelled_transition_data, reference, system, T):
    """All supported reference representations agree across fit and transform.

    Compare energies with nctpy's vector input to check the reference values
    reach the solver correctly in both time systems.
    """
    A, X = labelled_transition_data
    parameters = dict(T=T, system=system, normalize_A=False, transitions="undirected")
    inputs = dict(A=A, X=X.iloc[:2], xr=reference)
    estimator = Transitioner(**parameters).fit(**inputs)
    separate = estimator.transform(**inputs)
    combined = Transitioner(**parameters).fit_transform(**inputs)
    vector = estimator.transform(A=A, X=X.iloc[:2], xr=np.array([0.2, 0.4]))
    _, controls, errors = get_control_inputs(
        A_norm=A.to_numpy(), T=T, B=np.eye(2),
        x0=X.iloc[0].to_numpy(), xf=X.iloc[1].to_numpy(),
        system=system, rho=1.0, S=np.eye(2), xr=np.array([0.2, 0.4]),
    )
    assert estimator.n_nodes_ == 2
    assert separate.columns.equals(X.columns)
    np.testing.assert_allclose(separate.iloc[0], integrate_u(controls))
    np.testing.assert_allclose(estimator.get_errors()[0], errors)
    np.testing.assert_allclose(combined, separate)
    np.testing.assert_allclose(vector, separate)


@pytest.mark.parametrize("method", ["fit", "transform"])
@pytest.mark.parametrize(
    ("reference", "error", "message"),
    [
        (np.ones((2, 2)), ValueError, "xr must represent exactly one state"),
        (np.ones((2, 1)), ValueError, "xr must represent exactly one state"),
        ([[0.2], [0.4]], ValueError, "xr must represent exactly one state"),
        (pd.DataFrame(np.ones((2, 2))), ValueError, "xr must represent exactly one state"),
        (np.empty((0, 2)), ValueError, "xr must represent exactly one state"),
        (np.array(0.2), ValueError, "xr must be 1D or 2D"),
        (np.ones((1, 2, 1)), ValueError, "xr must be 1D or 2D"),
        ([np.nan, 0.4], ValueError, "xr must contain only finite values"),
        (np.array([[0.2, np.inf]]), ValueError, "xr must contain only finite values"),
        (["left", "right"], TypeError, "xr must contain numeric values"),
        (None, ValueError, "xr must be provided when energy_type"),
        (0.2, TypeError, "xr must be a 1D/2D array-like object"),
        ({"left": 0.2}, TypeError, "xr must be a 1D/2D array-like object"),
        ("invalid", ValueError, "xr must be 'zero', 'x0', 'xf', 'midpoint'"),
        (np.ones(3), ValueError, "same number of nodes"),
        (np.ones((1, 3)), ValueError, "same number of nodes"),
    ],
)
def test_invalid_xr_rejected_in_fit_and_transform(transition_data, method, reference, error, message):
    """Reject invalid references consistently before calling the solver."""
    A, X = transition_data
    estimator = Transitioner(T=0.002)
    if method == "transform":
        estimator.fit(A=A, X=X)
    with pytest.raises(error, match=message):
        getattr(estimator, method)(A=A, X=X, xr=reference)


@pytest.mark.parametrize(
    "reference",
    [
        pytest.param(np.array([0.2, 0.4]), id="vector"),
        pytest.param(np.array([[0.2, 0.4]]), id="single-row-array"),
        pytest.param([[0.2, 0.4]], id="single-row-list"),
        pytest.param(pd.DataFrame([[0.2, 0.4]]), id="single-row-dataframe"),
    ],
)
def test_xr_braincontrol_orientation_is_normalized_for_nctpy(transition_data, reference):
    """Convert braincontrol's (1, N) reference to nctpy's (N,) vector."""
    A, X = transition_data
    estimator = Transitioner(T=0.002).fit(A=A, X=X, xr=reference)
    states, counts, _, _ = estimator._process_states({"xr": reference})
    assert states["xr"].shape == (2,)
    assert counts["xr"] == 2
    np.testing.assert_array_equal(states["xr"], [0.2, 0.4])


@pytest.mark.parametrize("container", [pd.Series, pd.DataFrame])
def test_xr_preserves_node_labels(transition_data, container):
    """A reference alone can establish labels and enforce them at transform."""
    A, X = transition_data
    nodes = pd.Index(["left", "right"], name="node")
    reference = pd.Series([0.2, 0.4], index=nodes, name="reference")
    if container is pd.DataFrame:
        reference = reference.to_frame().T
    estimator = Transitioner(T=0.002).fit(A=A, X=X, xr=reference)
    assert estimator.node_labels_.equals(nodes)
    energy = estimator.transform(A=A, X=X, xr=reference)
    assert energy.columns.equals(nodes)

    reordered = reference.iloc[::-1] if container is pd.Series else reference.iloc[:, ::-1]
    with pytest.raises(ValueError, match="do not match the node labels established during fit"):
        estimator.transform(A=A, X=X, xr=reordered)


@pytest.mark.parametrize("method", ["fit", "transform"])
def test_xr_node_labels_must_match_other_inputs(labelled_transition_data, method):
    """Reference labels must agree with matrix and transition-state labels."""
    A, X = labelled_transition_data
    estimator = Transitioner(T=0.002)
    if method == "transform":
        estimator.fit(A=A, X=X)
    reference = pd.DataFrame([[0.2, 0.4]], columns=["right", "left"])
    with pytest.raises(ValueError, match="Node labels must match across all inputs"):
        getattr(estimator, method)(A=A, X=X, xr=reference)


def test_process_states_normalizes_xr_before_transform(transition_data):
    """Normalize the single row while preserving its node and state metadata."""
    A, X = transition_data
    estimator = Transitioner(T=0.002).fit(A=A, X=X)
    reference = pd.DataFrame([[0.2, 0.4]], columns=["left", "right"], index=["reference"])
    states, counts, nodes, labels = estimator._process_states({"xr": reference})
    assert states["xr"].shape == (2,)
    np.testing.assert_array_equal(states["xr"], [0.2, 0.4])
    assert counts["xr"] == 2
    assert nodes["xr"].equals(reference.columns)
    assert labels["xr"].equals(reference.index)


@pytest.mark.parametrize("reference", ["zero", "x0", "xf", "midpoint"])
def test_symbolic_xr_matches_explicit_state(transition_data, reference):
    """Keep symbolic references equivalent to the corresponding node vector."""
    A, X = transition_data
    explicit = {
        "zero": np.zeros(2),
        "x0": X[0],
        "xf": X[1],
        "midpoint": (X[0] + X[1]) / 2,
    }
    estimator = Transitioner(T=0.01, transitions="paired")
    symbolic = estimator.fit_transform(A=A, X0=X[0], Xf=X[1], xr=reference)
    numeric = estimator.transform(A=A, X0=X[0], Xf=X[1], xr=explicit[reference])
    np.testing.assert_allclose(symbolic, numeric)


@pytest.mark.parametrize("reference", [np.array([True]), np.array([[True]])])
def test_xr_single_node_boolean_state(reference):
    """Keep a one-node reference as a vector and accept boolean state values."""
    estimator = Transitioner(T=0.01, transitions="paired")
    inputs = dict(A=np.array([[-1.0]]), X0=[0.0], Xf=[1.0])
    energy = estimator.fit_transform(**inputs, xr=reference)
    expected = estimator.transform(**inputs, xr=[1.0])
    assert energy.shape == (1, 1)
    np.testing.assert_allclose(energy, expected)


####################
## Minimal energy
####################

@pytest.mark.parametrize(("system", "T"), [("continuous", 0.01), ("discrete", 3)])
def test_minimal_energy_matches_nctpy(transition_data, system, T):
    """Public None values still resolve to rho=1, zero S, and xr='zero'."""
    A, X = transition_data
    estimator = Transitioner(
        T=T, system=system, energy_type="minimal", rho=None,
        normalize_A=False, transitions="undirected",
    )
    inputs = dict(A=A, X=X[:2], S=None, xr=None)
    combined = estimator.fit_transform(**inputs)
    separate = estimator.transform(**inputs)
    _, controls, errors = get_control_inputs(
        A_norm=A, T=T, B=np.eye(2), x0=X[0], xf=X[1],
        system=system, rho=1.0, S=np.zeros((2, 2)), xr="zero",
    )
    assert estimator.rho is None
    assert estimator.rho_ == 1.0
    assert np.all(np.isfinite(combined))
    np.testing.assert_allclose(combined[0], integrate_u(controls))
    np.testing.assert_allclose(estimator.get_errors()[0], errors)
    np.testing.assert_allclose(separate, combined)


def test_minimal_energy_requires_rho_none(transition_data):
    """Minimal energy must not silently accept the default optimal rho."""
    A, X = transition_data
    with pytest.raises(ValueError, match="rho must be None"):
        Transitioner(T=0.002, energy_type="minimal").fit(A=A, X=X, S=None, xr=None)


@pytest.mark.parametrize("method", ["fit", "transform"])
@pytest.mark.parametrize(
    ("parameter", "value"),
    [("S", "identity"), ("S", np.zeros((2, 2))), ("xr", "zero"), ("xr", np.zeros(2))],
)
def test_minimal_energy_requires_S_and_xr_none(transition_data, method, parameter, value):
    """Only None is accepted publicly, even for the internal minimal defaults."""
    A, X = transition_data
    estimator = Transitioner(T=0.002, energy_type="minimal", rho=None)
    inputs = dict(A=A, X=X, S=None, xr=None)
    if method == "transform":
        estimator.fit(**inputs)
    inputs[parameter] = value
    with pytest.raises(ValueError, match=f"{parameter} must be None"):
        getattr(estimator, method)(**inputs)


####################
## Transition strategies
####################

@pytest.mark.parametrize(
    ("strategy", "n_transitions"),
    [("directed", 6), ("undirected", 3), ("directed_with_self", 9), ("self", 3)],
)
def test_X_transition_strategies(transition_data, strategy, n_transitions):
    """Test all valid transition strategies for a multi-state ``X``.
    
    Each strategy should complete successfully and produce the expected number
    of transitions while retaining one energy value per node."""
    A, X = transition_data
    energy = Transitioner(T=0.002, transitions=strategy).fit_transform(A=A, X=X)
    assert energy.shape == (n_transitions, 2)


def test_single_state_allows_self(transition_data):
    """Test the valid transition strategy for a single state.
    
    A one-state input should accept ``transitions='self'`` and produce exactly
    one self-transition rather than an empty transition set."""
    A, X = transition_data
    energy = Transitioner(T=0.002, transitions="self").fit_transform(A=A, X=X[0])
    assert energy.shape == (1, 2)


@pytest.mark.parametrize("strategy", ["directed", "undirected", "directed_with_self"])
def test_single_state_rejects_non_self_strategies(transition_data, strategy):
    """Test rejection of invalid strategies for a single state.
    
    Strategies requiring transitions between distinct states must raise a clear
    error when ``X`` contains only one state."""
    A, X = transition_data
    with pytest.raises(ValueError, match="not compatible with 1 state"):
        Transitioner(T=0.002, transitions=strategy).fit(A=A, X=X[0])


@pytest.mark.parametrize(("strategy", "n_transitions"), [("all_to_all", 4), ("paired", 2)])
def test_X0_Xf_transition_strategies(transition_data, strategy, n_transitions):
    """Test valid strategies for separate ``X0`` and ``Xf`` state sets.
    
    Both all-to-all and paired modes should produce the expected number of
    transitions for compatible initial and final state inputs."""
    A, X = transition_data
    energy = Transitioner(T=0.002, transitions=strategy).fit_transform(
        A=A, X0=X[:2], Xf=X[1:]
    )
    assert energy.shape == (n_transitions, 2)


def test_paired_requires_equal_state_counts(transition_data):
    """Test the cardinality requirement of paired transitions.
    
    ``transitions='paired'`` requires a one-to-one mapping between ``X0`` and
    ``Xf`` and therefore must reject state sets with different sizes."""
    A, X = transition_data
    with pytest.raises(ValueError, match="same number of states"):
        Transitioner(T=0.002, transitions="paired").fit(A=A, X0=X[:2], Xf=X[:1])


def test_X_and_X0_Xf_are_mutually_exclusive(transition_data):
    """Test the mutually exclusive state-input configurations.
    
    The estimator must accept either the combined ``X`` representation or the
    separate ``X0``/``Xf`` representation, but never both in one call."""
    A, X = transition_data
    with pytest.raises(ValueError, match="not both"):
        Transitioner(T=0.002).fit(A=A, X=X, X0=X[:1], Xf=X[1:2])


####################
## Node schema
####################

def test_transform_requires_fitted_node_count(transition_data):
    """Test transform-time node counts against the fitted schema.
    
    After fitting on a two-node system, transform inputs describing a different
    number of nodes must be rejected before transition computation."""
    A, X = transition_data
    estimator = Transitioner(T=0.002).fit(A=A, X=X)
    with pytest.raises(ValueError, match="fitted schema contains 2 nodes"):
        estimator.transform(A=np.eye(3), X=np.ones((2, 3)))


def test_transform_rejects_unlabelled_schema_after_labelled_fit(labelled_transition_data):
    """Test strict node-label consistency after a labelled fit.
    
    If fitting established node labels, later transform inputs without labels
    must be rejected even when their number of nodes is otherwise compatible."""
    A, X = labelled_transition_data
    estimator = Transitioner(T=0.002).fit(A=A, X=X)
    with pytest.raises(ValueError, match="no node labels were provided during transform"):
        estimator.transform(A=A.to_numpy(), X=X.to_numpy())


def test_transform_rejects_labels_after_unlabelled_fit(transition_data):
    """Test strict node-label consistency after an unlabelled fit.
    
    If fitting established no node labels, transform must reject newly labelled
    inputs rather than silently changing the fitted node schema."""
    A, X = transition_data
    estimator = Transitioner(T=0.002).fit(A=A, X=X)
    labels = pd.Index(["left", "right"], name="node")
    A_df = pd.DataFrame(A, index=labels, columns=labels)
    X_df = pd.DataFrame(X, columns=labels)
    with pytest.raises(ValueError, match="no node labels were established during fit"):
        estimator.transform(A=A_df, X=X_df)


####################
## Labels and output metadata
####################

def test_dataframe_state_labels_become_transition_labels(labelled_transition_data):
    """Test inference of transition labels from DataFrame state labels.
    
    State labels from the DataFrame index should be combined according to the
    computed transitions and used as the energy DataFrame index."""
    A, X = labelled_transition_data
    energy = Transitioner(T=0.002, transitions="undirected").fit_transform(A=A, X=X)
    assert isinstance(energy, pd.DataFrame)
    assert energy.index.name == "transition"
    assert energy.index.tolist() == [
        ("rest", "task"), ("rest", "recovery"), ("task", "recovery")
    ]
    assert energy.columns.equals(X.columns)


def test_series_name_becomes_single_state_label():
    """Test state-label inference for a one-state pandas Series.
    
    The Series name should represent the state label and therefore label both
    the source and target of its single self-transition."""
    nodes = pd.Index(["left", "right"], name="node")
    A = pd.DataFrame([[-1.0, 0.1], [0.1, -1.2]], index=nodes, columns=nodes)
    X = pd.Series([1.0, 0.0], index=nodes, name="rest")
    energy = Transitioner(T=0.002, transitions="self").fit_transform(A=A, X=X)
    assert energy.index.tolist() == [("rest", "rest")]


def test_multiindex_state_labels_are_preserved():
    """Test hierarchical state metadata in transition labels.
    
    MultiIndex level names and values should be preserved when source and target
    state labels are combined into transition-level output metadata."""
    nodes = pd.Index(["left", "right"], name="node")
    states = pd.MultiIndex.from_tuples(
        [("baseline", "rest"), ("active", "task")],
        names=["condition", "state"],
    )
    A = pd.DataFrame([[-1.0, 0.1], [0.1, -1.2]], index=nodes, columns=nodes)
    X = pd.DataFrame([[1.0, 0.0], [0.0, 1.0]], index=states, columns=nodes)
    energy = Transitioner(T=0.002, transitions="directed").fit_transform(A=A, X=X)
    assert isinstance(energy.index, pd.MultiIndex)
    assert energy.index.names == ["condition", "state"]
    assert energy.index.tolist() == [
        (("baseline", "active"), ("rest", "task")),
        (("active", "baseline"), ("task", "rest")),
    ]


def test_X0_Xf_can_have_different_state_labels():
    """Test distinct state labels for initial and final state sets.
    
    ``X0`` and ``Xf`` may contain different label values; paired transition
    labels should combine the corresponding source and target labels correctly."""
    nodes = pd.Index(["left", "right"], name="node")
    A = pd.DataFrame([[-1.0, 0.1], [0.1, -1.2]], index=nodes, columns=nodes)
    X0 = pd.DataFrame([[1.0, 0.0], [0.0, 1.0]], index=["rest", "task"], columns=nodes)
    Xf = pd.DataFrame([[0.5, 0.5], [0.25, 0.75]], index=["recovery", "followup"], columns=nodes)
    energy = Transitioner(T=0.002, transitions="paired").fit_transform(A=A, X0=X0, Xf=Xf)
    assert energy.index.tolist() == [("rest", "recovery"), ("task", "followup")]


def test_X0_Xf_reject_mixed_state_labels(transition_data):
    """Test consistent label availability across ``X0`` and ``Xf``.
    
    Separate state sets must either both provide state labels or both be
    unlabelled so that transition labels can be constructed unambiguously."""
    A, X = transition_data
    X0 = pd.DataFrame(X[:1], index=["rest"])
    with pytest.raises(ValueError, match="both provide state labels or both be unlabeled"):
        Transitioner(T=0.002, transitions="paired").fit_transform(A=A, X0=X0, Xf=X[1:2])


def test_unlabelled_energy_output_is_ndarray(transition_data):
    """Test the energy output type when no metadata is available.
    
    With entirely unlabelled NumPy inputs, the estimator should return a plain
    NumPy array rather than introducing artificial pandas labels."""
    A, X = transition_data
    energy = Transitioner(T=0.002, transitions="undirected").fit_transform(A=A, X=X)
    assert isinstance(energy, np.ndarray)
    assert energy.shape == (3, 2)


####################
## Stored results and trajectories
####################

def test_get_errors_returns_copy(transition_data):
    """Test access to numerical errors from the latest transform.
    
    ``get_errors`` should return the expected error array as a defensive copy so
    that modifying the returned object cannot mutate estimator state."""
    A, X = transition_data
    estimator = Transitioner(T=0.002, transitions="undirected")
    estimator.fit_transform(A=A, X=X)
    errors = estimator.get_errors()
    assert errors.shape == (3, 2)
    errors[:] = np.nan
    assert not np.isnan(estimator.errors_).all()


def test_get_errors_requires_transform(transition_data):
    """Test that numerical errors are unavailable before transformation.
    
    Fitting alone does not compute transition trajectories, so ``get_errors``
    must raise ``NotFittedError`` until a transform has been performed."""
    A, X = transition_data
    estimator = Transitioner(T=0.002).fit(A=A, X=X)
    with pytest.raises(NotFittedError):
        estimator.get_errors()


def test_trajectory_getters_return_none_when_storage_disabled(transition_data):
    """Test trajectory getters when trajectory retention is disabled.
    
    A successful transform should not retain state or control trajectories
    unless the corresponding constructor options explicitly request storage."""
    A, X = transition_data
    estimator = Transitioner(T=0.002, transitions="undirected")
    estimator.fit_transform(A=A, X=X)
    assert estimator.get_state_trajectories() is None
    assert estimator.get_control_trajectories() is None


def test_trajectory_getters_return_xarray(labelled_transition_data):
    """Test labelled xarray output from the trajectory getters.
    
    When trajectory storage is enabled, both getters should expose arrays with
    time, node, and transition dimensions and preserve available coordinates."""
    A, X = labelled_transition_data
    estimator = Transitioner(
        T=0.002,
        transitions="undirected",
        store_state_trajectories=True,
        store_control_trajectories=True,
    )
    estimator.fit_transform(A=A, X=X)
    state = estimator.get_state_trajectories()
    control = estimator.get_control_trajectories()
    assert isinstance(state, xr.DataArray)
    assert isinstance(control, xr.DataArray)
    assert state.dims == ("time", "node", "transition")
    assert control.dims == ("time", "node", "transition")
    assert state.shape[1:] == (2, 3)
    assert control.shape[1:] == (2, 3)
    assert state.coords["node"].values.tolist() == ["left", "right"]
    assert state.coords["transition"].values.tolist() == [
        ("rest", "task"), ("rest", "recovery"), ("task", "recovery")
    ]


####################
## Neuroimaging
####################

@pytest.mark.parametrize("shape", [(2, 1, 1), (2, 1, 1, 1)])
def test_xr_single_nifti_image_matches_array(transition_data, shape):
    """Both 3D and single-volume 4D references agree with a node vector."""
    A, X = transition_data
    reference = np.array([0.2, 0.4])
    image = nib.Nifti1Image(reference.reshape(shape), np.eye(4))
    labels_img = nib.Nifti1Image(
        np.array([1, 2], dtype=np.int16).reshape(2, 1, 1), np.eye(4)
    )
    masker = NiftiLabelsMasker(
        labels_img=labels_img, standardize=None, reports=False,
        keep_masked_labels=False,
    )
    estimator = Transitioner(T=0.01, transitions="paired", masker=masker)
    inputs = dict(A=A, X0=X[0], Xf=X[1])
    image_energy = estimator.fit_transform(**inputs, xr=image)
    array_energy = estimator.transform(**inputs, xr=reference)
    np.testing.assert_allclose(image_energy, array_energy)


@pytest.mark.parametrize("method", ["fit", "transform"])
def test_xr_rejects_multiple_nifti_states(transition_data, method):
    """Reject multiple reference volumes before attempting to mask them."""
    A, X = transition_data
    image = nib.Nifti1Image(np.ones((2, 1, 1, 2)), np.eye(4))
    estimator = Transitioner(T=0.002)
    if method == "transform":
        estimator.fit(A=A, X=X)
    with pytest.raises(ValueError, match="xr is Niimg-like, it must represent exactly one state"):
        getattr(estimator, method)(A=A, X=X, xr=image)


def test_transitioner_masks_4d_nifti_image(transition_data):
    """Test end-to-end processing of Niimg-like state input.
    
    Applying a compatible Nilearn labels masker to a 4D image should yield the
    same transition energies as supplying the equivalent node-level array."""
    A, X = transition_data
    labels_img = nib.Nifti1Image(
        np.array([1, 2], dtype=np.int16).reshape(2, 1, 1), np.eye(4)
    )
    states_img = nib.Nifti1Image(X.T.reshape(2, 1, 1, 3), np.eye(4))
    masker = NiftiLabelsMasker(
        labels_img=labels_img,
        standardize=None,
        reports=False,
        keep_masked_labels=False,
    )
    image_energy = Transitioner(
        T=0.002, transitions="undirected", masker=masker
    ).fit_transform(A=A, X=states_img)
    array_energy = Transitioner(
        T=0.002, transitions="undirected"
    ).fit_transform(A=A, X=X)
    np.testing.assert_allclose(image_energy, array_energy)


def test_image_input_requires_masker(transition_data):
    """Test the masker requirement for Niimg-like state input.
    
    Image data cannot be converted to node-level states without a compatible
    Nilearn masker, so fitting must reject such input when no masker is given."""
    A, X = transition_data
    image = nib.Nifti1Image(X.T.reshape(2, 1, 1, 3), np.eye(4))
    with pytest.raises(TypeError, match="masker must be a nilearn BaseMasker"):
        Transitioner(T=0.002, transitions="undirected").fit(A=A, X=image)

def test_user_masker_is_not_fitted_in_place(transition_data):
    """Test that the estimator does not mutate the user-provided masker.
    
    Niimg processing should clone and fit the masker internally, leaving the
    original masker object unfitted and reusable by the caller."""
    A, X = transition_data
    labels_img = nib.Nifti1Image(
        np.array([1, 2], dtype=np.int16).reshape(2, 1, 1), np.eye(4)
    )
    image = nib.Nifti1Image(X.T.reshape(2, 1, 1, 3), np.eye(4))
    masker = NiftiLabelsMasker(
        labels_img=labels_img,
        standardize=None,
        reports=False,
        keep_masked_labels=False,
    )
    Transitioner(T=0.002, transitions="undirected", masker=masker).fit(A=A, X=image)
    assert not hasattr(masker, "n_elements_")
