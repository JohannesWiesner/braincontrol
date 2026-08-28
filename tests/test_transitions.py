import nibabel as nib
import numpy as np
import pandas as pd
import pytest
from nctpy.energies import get_control_inputs, integrate_u
from nctpy.utils import matrix_normalization
from nilearn._utils.cache_mixin import CacheMixin
from nilearn.maskers import NiftiLabelsMasker
from sklearn.base import clone

import braincontrol.transitions as transitions
from braincontrol.transitions import (
    Transitioner,
    _coerce_labels,
    _resolve_state_input,
    _set_transition_order,
    _state_transition_index,
    _validate_transition_inputs,
    _validate_transition_order,
    get_transition_energy,
    get_transition_trajectories,
)


@pytest.fixture
def transition_data():
    """Provide a small adjacency matrix and three states for transition tests."""
    adjacency = np.array([[-1.0, 0.1], [0.1, -1.2]])
    states = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.25]])
    return adjacency, states


@pytest.mark.parametrize(
    ("order", "expected"),
    [
        ("combinations", [(0, 0, 1), (1, 0, 2), (2, 1, 2)]),
        (
            "permutations",
            [
                (0, 0, 1),
                (1, 0, 2),
                (2, 1, 0),
                (3, 1, 2),
                (4, 2, 0),
                (5, 2, 1),
            ],
        ),
        (
            "product",
            [
                (0, 0, 0),
                (1, 0, 1),
                (2, 0, 2),
                (3, 1, 0),
                (4, 1, 1),
                (5, 1, 2),
                (6, 2, 0),
                (7, 2, 1),
                (8, 2, 2),
            ],
        ),
        ("stability", [(0, 0, 0), (1, 1, 1), (2, 2, 2)]),
    ],
)
def test_set_transition_order(order, expected):
    """Check that every transition-order mode returns the expected state pairs."""
    count, indices = _set_transition_order(3, order)

    assert count == len(expected)
    assert indices == expected


def test_set_transition_order_rejects_invalid_order():
    """Check that unknown transition-order names are rejected."""
    with pytest.raises(ValueError, match="order must be one of"):
        _set_transition_order(3, "invalid")


def test_validate_transition_order_rejects_single_state_pairs():
    """Check that pairwise orders require at least two input states."""
    for order in ("combinations", "permutations"):
        with pytest.raises(ValueError, match="requires at least two states"):
            _validate_transition_order(1, order)


@pytest.mark.parametrize("order", ["product", "stability"])
def test_validate_transition_order_accepts_single_state_self_transitions(order):
    """Check that self-transition orders remain valid for a single state."""
    assert _validate_transition_order(1, order) == order


def test_transition_trajectories_match_nctpy(transition_data):
    """Check the low-level trajectory helper against a direct nctpy call."""
    adjacency, states = transition_data
    horizon = 0.002
    B = np.eye(2)
    S = np.eye(2)

    trajectories, controls, errors = get_transition_trajectories(
        A=adjacency,
        X=states,
        T=horizon,
        B=B,
        rho=1.0,
        S=S,
        order="permutations",
        system="continuous",
    )

    expected_x, expected_u, expected_error = get_control_inputs(
        A_norm=adjacency,
        T=horizon,
        B=B,
        x0=states[0],
        xf=states[1],
        system="continuous",
        rho=1.0,
        S=S,
        xr="xf",
    )

    assert trajectories.shape == (3, 2, 6)
    assert controls.shape == (3, 2, 6)
    assert errors.shape == (6, 2)
    np.testing.assert_allclose(trajectories[:, :, 0], expected_x)
    np.testing.assert_allclose(controls[:, :, 0], expected_u)
    np.testing.assert_allclose(errors[0], expected_error)


def test_transition_energy_matches_nctpy_integration(transition_data):
    """Check that transition energies equal nctpy's integrated controls."""
    adjacency, states = transition_data
    _, controls, _ = get_transition_trajectories(
        A=adjacency,
        X=states[:2],
        T=0.002,
        B=np.eye(2),
        rho=1.0,
        S=np.eye(2),
        order="permutations",
    )

    energies = get_transition_energy(controls)

    assert energies.shape == (2, 2)
    np.testing.assert_allclose(energies[0], integrate_u(controls[:, :, 0]))


@pytest.mark.parametrize(
    ("energy_type", "rho", "S", "message"),
    [
        ("invalid", 1.0, "identity", "energy_type must be one of"),
        ("minimal", 1.0, None, "rho and S must both be None"),
        ("minimal", None, "identity", "rho and S must both be None"),
        ("optimal", None, None, "rho and S must both be provided"),
        ("optimal", 1.0, None, "rho and S must both be provided"),
    ],
)
def test_validate_transition_inputs_checks_energy_parameters(
    transition_data, energy_type, rho, S, message
):
    """Check consistency requirements between energy type, rho, and S."""
    adjacency, _ = transition_data

    with pytest.raises(ValueError, match=message):
        _validate_transition_inputs(
            adjacency,
            0.002,
            "identity",
            rho,
            S,
            energy_type,
            "continuous",
            "zero",
            "scipy",
            True,
            1,
        )


@pytest.mark.parametrize("rho", [0.0, -0.1, 1.1, np.inf, np.nan])
def test_validate_transition_inputs_checks_optimal_rho(transition_data, rho):
    """Check that optimal-control rho lies in the supported finite interval."""
    adjacency, _ = transition_data

    with pytest.raises(ValueError, match="between 0 and 1"):
        _validate_transition_inputs(
            adjacency,
            0.002,
            "identity",
            rho,
            "identity",
            "optimal",
            "continuous",
            "zero",
            "scipy",
            True,
            1,
        )


def test_resolve_state_matrix_returns_dataframe(transition_data):
    """Check that a NumPy state matrix resolves to a two-dimensional DataFrame."""
    _, states = transition_data

    resolved, X_type, xr, xr_type = _resolve_state_input(X=states)

    assert isinstance(resolved, pd.DataFrame)
    np.testing.assert_array_equal(resolved.to_numpy(), states)
    assert X_type == "tabular_like"
    assert xr == "xf"
    assert xr_type == "named"


def test_resolve_dataframe_preserves_labels(transition_data):
    """Check that DataFrame state and node labels survive state resolution."""
    _, states = transition_data
    frame = pd.DataFrame(
        states,
        index=pd.Index(["rest", "task", "recovery"], name="state"),
        columns=pd.Index(["A", "B"], name="node"),
    )

    resolved, _, _, _ = _resolve_state_input(X=frame)

    assert resolved is not frame
    assert resolved.index.equals(frame.index)
    assert resolved.columns.equals(frame.columns)


def test_resolve_numpy_endpoints_returns_two_state_dataframe(transition_data):
    """Check that separate NumPy endpoints become two rows in a DataFrame."""
    _, states = transition_data

    resolved, _, _, _ = _resolve_state_input(x0=states[0], xf=states[1])

    assert resolved.shape == (2, 2)
    np.testing.assert_array_equal(resolved.to_numpy(), states[:2])


def test_resolve_series_endpoints_preserves_node_labels(transition_data):
    """Check that Series endpoint indices become DataFrame node labels."""
    _, states = transition_data
    node_labels = pd.Index(["left", "right"], name="node")
    x0 = pd.Series(states[0], index=node_labels)
    xf = pd.Series(states[1], index=node_labels)

    resolved, _, _, _ = _resolve_state_input(x0=x0, xf=xf)

    assert resolved.shape == (2, 2)
    assert resolved.columns.equals(node_labels)


def test_resolve_state_input_requires_one_complete_input_mode(transition_data):
    """Check that X and x0/xf are mutually exclusive and endpoints are complete."""
    _, states = transition_data

    with pytest.raises(ValueError, match="Provide either X or both x0 and xf"):
        _resolve_state_input()

    with pytest.raises(ValueError, match="x0 and xf must be provided together"):
        _resolve_state_input(x0=states[0])

    with pytest.raises(ValueError, match="Provide either X or x0 and xf, not both"):
        _resolve_state_input(X=states, x0=states[0], xf=states[1])


def test_resolve_state_input_rejects_mixed_endpoint_types(transition_data):
    """Check that x0 and xf must use the same endpoint representation."""
    _, states = transition_data
    xf = pd.Series(states[1])

    with pytest.raises(TypeError, match="same input type"):
        _resolve_state_input(x0=states[0], xf=xf)


def test_resolve_state_input_rejects_invalid_endpoint_shapes(transition_data):
    """Check that tabular endpoints are one-dimensional and equally sized."""
    _, states = transition_data

    with pytest.raises(ValueError, match="one-dimensional states"):
        _resolve_state_input(x0=states[[0]], xf=states[[1]])

    with pytest.raises(ValueError, match="same number of nodes"):
        _resolve_state_input(x0=states[0], xf=np.ones(3))


def test_resolve_niimg_X_is_always_4d():
    """Check that a single 3D image resolves to a singleton 4D image."""
    image = nib.Nifti1Image(np.ones((2, 1, 1)), np.eye(4))

    resolved, X_type, _, _ = _resolve_state_input(X=image)

    assert resolved.ndim == 4
    assert resolved.shape == (2, 1, 1, 1)
    assert X_type == "niimg_like"


def test_resolve_niimg_endpoints_returns_two_volume_image():
    """Check that two image endpoints resolve to a 4D image with two states."""
    x0 = nib.Nifti1Image(np.ones((2, 1, 1)), np.eye(4))
    xf = nib.Nifti1Image(np.zeros((2, 1, 1)), np.eye(4))

    resolved, _, _, _ = _resolve_state_input(x0=x0, xf=xf)

    assert resolved.ndim == 4
    assert resolved.shape == (2, 1, 1, 2)


def test_resolve_niimg_endpoint_must_contain_one_state():
    """Check that an image endpoint cannot itself contain multiple states."""
    x0 = nib.Nifti1Image(np.ones((2, 1, 1, 2)), np.eye(4))
    xf = nib.Nifti1Image(np.ones((2, 1, 1)), np.eye(4))

    with pytest.raises(ValueError, match="exactly one state"):
        _resolve_state_input(x0=x0, xf=xf)


def test_coerce_plain_state_labels_creates_named_index():
    """Check that plain state labels become an Index named 'state'."""
    labels = _coerce_labels(["rest", "task"], 2, "state_labels")

    assert labels.equals(pd.Index(["rest", "task"], name="state"))


def test_coerce_plain_node_labels_creates_named_index():
    """Check that plain node labels become an Index named 'node'."""
    labels = _coerce_labels(["A", "B"], 2, "node_labels")

    assert labels.equals(pd.Index(["A", "B"], name="node"))


def test_coerce_labels_preserves_existing_index_name():
    """Check that an existing Index keeps its user-provided name."""
    original = pd.Index(["rest", "task"], name="condition")

    labels = _coerce_labels(original, 2, "state_labels")

    assert labels.equals(original)
    assert labels.name == "condition"


def test_coerce_labels_requires_named_multiindex_levels():
    """Check that every level of a supplied MultiIndex has a name."""
    labels = pd.MultiIndex.from_tuples(
        [("rest", "young"), ("task", "old")],
        names=["condition", None],
    )

    with pytest.raises(ValueError, match="must have a name"):
        _coerce_labels(labels, 2, "state_labels")


def test_coerce_labels_validates_length():
    """Check that the number of labels matches the expected axis length."""
    with pytest.raises(ValueError, match="must contain 2 values"):
        _coerce_labels(["A"], 2, "node_labels")


def test_state_transition_index_from_named_index():
    """Check that regular state labels become source-target tuple labels."""
    labels = pd.Index(["rest", "task"], name="condition")

    result = _state_transition_index(labels, "permutations")

    assert result.name == "condition"
    assert result.tolist() == [("rest", "task"), ("task", "rest")]


def test_state_transition_index_from_multiindex():
    """Check that MultiIndex levels are preserved as transition-label levels."""
    labels = pd.MultiIndex.from_tuples(
        [("rest", "young"), ("task", "old")],
        names=["condition", "group"],
    )

    result = _state_transition_index(labels, "permutations")

    assert isinstance(result, pd.MultiIndex)
    assert result.names == ["condition", "group"]
    assert result.tolist() == [
        (("rest", "task"), ("young", "old")),
        (("task", "rest"), ("old", "young")),
    ]


def test_transitioner_is_public_transition_class():
    """Check that Transitioner is exported as the public estimator."""
    assert "Transitioner" in transitions.__all__
    assert not hasattr(transitions, "TransitionTransformer")
    assert not hasattr(transitions, "StateTransitionTransformer")


def test_transitioner_inherits_nilearn_cache_mixin():
    """Check that Transitioner exposes Nilearn's caching behavior."""
    assert issubclass(Transitioner, CacheMixin)


def test_transitioner_fits_array_states(transition_data):
    """Check fitted matrix, state-count, node-count, and node-label metadata."""
    adjacency, states = transition_data
    transitioner = Transitioner(A=adjacency, T=0.002).fit(states)

    assert transitioner.n_states_in_ == 3
    assert transitioner.n_nodes_in_ == 2
    assert transitioner.n_features_in_ == 2
    assert transitioner.X_type_ == "tabular_like"
    assert transitioner.node_labels_.equals(pd.RangeIndex(2))
    np.testing.assert_array_equal(transitioner.A_, adjacency)


def test_transitioner_normalizes_adjacency_during_fit(transition_data):
    """Check that fit stores the normalized adjacency matrix separately."""
    adjacency, states = transition_data

    transitioner = Transitioner(A=adjacency, T=0.002).fit(states)

    expected = matrix_normalization(
        adjacency,
        system="continuous",
        c=1,
    )
    np.testing.assert_array_equal(transitioner.A_, adjacency)
    np.testing.assert_allclose(transitioner.A_norm_, expected)


def test_transitioner_can_use_pre_normalized_adjacency(transition_data):
    """Check that normalize_A=False stores an unchanged working adjacency."""
    adjacency, states = transition_data

    transitioner = Transitioner(
        A=adjacency,
        T=0.002,
        normalize_A=False,
    ).fit(states)

    np.testing.assert_array_equal(transitioner.A_norm_, adjacency)


@pytest.mark.parametrize("xr", ["zero", "x0", "xf", "midpoint"])
def test_transitioner_accepts_named_reference_states(transition_data, xr):
    """Check that every reference-state name supported by nctpy is retained."""
    adjacency, states = transition_data

    transitioner = Transitioner(A=adjacency, T=0.002).fit(
        pd.DataFrame(states),
        xr=xr,
    )

    assert transitioner.xr_ == xr


@pytest.mark.parametrize(
    "xr",
    [
        np.array([0.25, 0.75]),
        np.array([[0.25], [0.75]]),
        [0.25, 0.75],
        (0.25, 0.75),
        pd.Series([0.25, 0.75], index=["left", "right"]),
    ],
)
def test_transitioner_resolves_tabular_reference_state(transition_data, xr):
    """Check every tabular reference type becomes an nctpy column vector."""
    adjacency, states = transition_data

    transitioner = Transitioner(A=adjacency, T=0.002).fit(
        pd.DataFrame(states),
        xr=xr,
    )

    assert transitioner.xr_.shape == (2, 1)
    np.testing.assert_array_equal(transitioner.xr_.ravel(), [0.25, 0.75])
    if isinstance(xr, np.ndarray):
        assert not np.shares_memory(transitioner.xr_, xr)


@pytest.mark.parametrize(
    ("xr", "error", "message"),
    [
        (np.ones((1, 2)), ValueError, "one-dimensional or a column vector"),
        (np.ones((3, 1)), ValueError, "same number of nodes"),
        (np.array([[np.nan], [1.0]]), ValueError, "only finite"),
        (np.array([["a"], ["b"]]), TypeError, "numeric values"),
        ("unknown", ValueError, "xr must be one of"),
    ],
)
def test_transitioner_validates_reference_state(
    transition_data, xr, error, message
):
    """Check reference vectors and names before invoking nctpy."""
    adjacency, states = transition_data

    with pytest.raises(error, match=message):
        Transitioner(A=adjacency, T=0.002).fit(pd.DataFrame(states), xr=xr)


def test_transitioner_defaults_reference_state_to_xf(transition_data):
    """Check that the fitted reference defaults to each transition target."""
    adjacency, states = transition_data

    transitioner = Transitioner(A=adjacency, T=0.002).fit(
        pd.DataFrame(states)
    )

    assert transitioner.xr_ == "xf"


@pytest.mark.parametrize(
    "state_kwargs",
    [
        {"X": np.ones((2, 3))},
        {"x0": np.ones(3), "xf": np.zeros(3)},
    ],
)
def test_reference_state_must_match_state_node_count(state_kwargs):
    """Check xr against both matrix and separate-endpoint state inputs."""
    with pytest.raises(ValueError, match="same number of nodes as the state input"):
        Transitioner(A=np.eye(3), T=0.002).fit(
            **state_kwargs,
            xr=[0.25, 0.75],
        )


def test_transitioner_masks_3d_reference_image(transition_data):
    """Check that a 3D reference image becomes an nctpy node vector."""
    adjacency, states = transition_data
    labels = nib.Nifti1Image(
        np.array([1, 2], dtype=np.int16).reshape(2, 1, 1),
        np.eye(4),
    )
    reference = nib.Nifti1Image(
        np.array([0.25, 0.75]).reshape(2, 1, 1),
        np.eye(4),
    )
    masker = NiftiLabelsMasker(
        labels_img=labels,
        standardize=None,
        reports=False,
        keep_masked_labels=False,
    )

    transitioner = Transitioner(
        A=adjacency,
        T=0.002,
        masker=masker,
    ).fit(pd.DataFrame(states), xr=reference)

    assert transitioner.xr_.shape == (2, 1)
    np.testing.assert_allclose(transitioner.xr_.ravel(), [0.25, 0.75])

    image_energies = transitioner.transform(
        pd.DataFrame(states),
        order="combinations",
    )
    array_energies = Transitioner(
        A=adjacency,
        T=0.002,
    ).fit_transform(
        pd.DataFrame(states),
        xr=np.array([[0.25], [0.75]]),
        order="combinations",
    )
    np.testing.assert_allclose(image_energies, array_energies)


def test_image_reference_requires_masker(transition_data):
    """Check that image references cannot silently bypass parcellation."""
    adjacency, states = transition_data
    reference = nib.Nifti1Image(np.ones((2, 1, 1)), np.eye(4))

    with pytest.raises(ValueError, match="Image-like xr requires a masker"):
        Transitioner(A=adjacency, T=0.002).fit(
            pd.DataFrame(states),
            xr=reference,
        )


def test_reference_image_must_contain_one_state(transition_data):
    """Check that xr represents exactly one image reference state."""
    adjacency, states = transition_data
    reference = nib.Nifti1Image(np.ones((2, 1, 1, 2)), np.eye(4))

    with pytest.raises(ValueError, match="exactly one state"):
        Transitioner(A=adjacency, T=0.002).fit(
            pd.DataFrame(states),
            xr=reference,
        )


@pytest.mark.parametrize("normalize_A", [None, 1, "yes"])
def test_transitioner_requires_boolean_normalize_A(
    transition_data, normalize_A
):
    """Check that normalize_A accepts only boolean values."""
    adjacency, states = transition_data

    with pytest.raises(TypeError, match="normalize_A must be a boolean"):
        Transitioner(
            A=adjacency,
            T=0.002,
            normalize_A=normalize_A,
        ).fit(states)


@pytest.mark.parametrize("c", [None, True, "1"])
def test_transitioner_rejects_non_numeric_c(transition_data, c):
    """Check that c rejects non-numeric values and booleans."""
    adjacency, states = transition_data

    with pytest.raises(TypeError, match="c must be a real number"):
        Transitioner(A=adjacency, T=0.002, c=c).fit(states)


@pytest.mark.parametrize("c", [0, -1, np.inf, np.nan])
def test_transitioner_requires_positive_finite_c(transition_data, c):
    """Check that numeric c values are positive and finite."""
    adjacency, states = transition_data

    with pytest.raises(ValueError, match="c must be a positive finite number"):
        Transitioner(A=adjacency, T=0.002, c=c).fit(states)


def test_transitioner_rejects_minimal_energy_with_default_rho_and_S(
    transition_data,
):
    """Check that minimal energy requires explicit rho=None and S=None."""
    adjacency, states = transition_data
    transitioner = Transitioner(
        A=adjacency,
        T=0.002,
        energy_type="minimal",
    )

    with pytest.raises(ValueError, match="rho and S must both be None"):
        transitioner.fit(states)


def test_transitioner_resolves_minimal_energy_solver_parameters(
    transition_data,
):
    """Check that minimal energy resolves rho and S for the nctpy solver."""
    adjacency, states = transition_data

    transitioner = Transitioner(
        A=adjacency,
        T=0.002,
        energy_type="minimal",
        rho=None,
        S=None,
    ).fit(states, xr=None)

    assert transitioner.rho_ == 1.0
    np.testing.assert_array_equal(
        transitioner.S_,
        np.zeros_like(adjacency),
    )


def test_transitioner_requires_none_reference_for_minimal_energy(
    transition_data,
):
    """Check that minimal energy explicitly disables its reference state."""
    adjacency, states = transition_data
    transitioner = Transitioner(
        A=adjacency,
        T=0.002,
        energy_type="minimal",
        rho=None,
        S=None,
    )

    with pytest.raises(ValueError, match="xr must be None"):
        transitioner.fit(states)


def test_minimal_energy_with_none_reference_can_transform(transition_data):
    """Check the internal solver fallback for reference-free minimal energy."""
    adjacency, states = transition_data
    energies = Transitioner(
        A=adjacency,
        T=0.002,
        energy_type="minimal",
        rho=None,
        S=None,
    ).fit_transform(states, xr=None, order="combinations")

    assert energies.shape == (3, 2)


def test_transitioner_transform_returns_energy_dataframe(transition_data):
    """Check that transform returns transition-by-node energies as a DataFrame."""
    adjacency, states = transition_data
    transitioner = Transitioner(A=adjacency, T=0.002).fit(states)

    energies = transitioner.transform(states, order="combinations")

    assert isinstance(energies, pd.DataFrame)
    assert energies.shape == (3, 2)
    assert energies.columns.equals(transitioner.node_labels_)
    assert transitioner.get_errors().shape == (3, 2)


def test_transitioner_fit_transform_accepts_order(transition_data):
    """Check that fit_transform forwards transition order to transform."""
    adjacency, states = transition_data
    transitioner = Transitioner(A=adjacency, T=0.002)

    energies = transitioner.fit_transform(
        states,
        order="combinations",
    )

    assert energies.shape == (3, 2)


def test_transitioner_uses_fitted_node_labels_for_output(transition_data):
    """Check that fit-time node labels are reused for later transform output."""
    adjacency, states = transition_data
    node_labels = pd.MultiIndex.from_tuples(
        [("left", "A"), ("right", "B")],
        names=["hemisphere", "node"],
    )
    transitioner = Transitioner(A=adjacency, T=0.002).fit(
        states,
        node_labels=node_labels,
    )

    energies = transitioner.transform(
        states.copy(),
        order="combinations",
    )

    assert energies.columns.equals(node_labels)


def test_transitioner_infers_node_labels_from_dataframe(transition_data):
    """Check that DataFrame columns become fitted node labels."""
    adjacency, states = transition_data
    node_labels = pd.Index(["left", "right"], name="node")
    states_df = pd.DataFrame(states, columns=node_labels)

    transitioner = Transitioner(A=adjacency, T=0.002).fit(states_df)

    assert transitioner.node_labels_.equals(node_labels)


def test_transitioner_infers_state_labels_during_transform(transition_data):
    """Check that DataFrame row labels describe transitions for each transform call."""
    adjacency, states = transition_data
    state_labels = pd.Index(
        ["rest", "task", "recovery"],
        name="condition",
    )
    states_df = pd.DataFrame(states, index=state_labels)
    transitioner = Transitioner(A=adjacency, T=0.002).fit(states_df)

    energies = transitioner.transform(
        states_df,
        order="combinations",
    )

    assert energies.index.name == "condition"
    assert energies.index.tolist() == [
        ("rest", "task"),
        ("rest", "recovery"),
        ("task", "recovery"),
    ]


def test_transitioner_accepts_explicit_state_labels_at_transform(transition_data):
    """Check that transform-time state labels override inferred row labels."""
    adjacency, states = transition_data
    transitioner = Transitioner(A=adjacency, T=0.002).fit(states)

    energies = transitioner.transform(
        states,
        state_labels=["rest", "task", "recovery"],
        order="combinations",
    )

    assert energies.index.name == "state"
    assert energies.index.tolist() == [
        ("rest", "task"),
        ("rest", "recovery"),
        ("task", "recovery"),
    ]


def test_transitioner_accepts_multiindex_state_labels_at_transform(
    transition_data,
):
    """Check that hierarchical state metadata is preserved in transition labels."""
    adjacency, states = transition_data
    state_labels = pd.MultiIndex.from_tuples(
        [
            ("baseline", "rest"),
            ("active", "task"),
            ("baseline", "recovery"),
        ],
        names=["condition", "state"],
    )
    transitioner = Transitioner(A=adjacency, T=0.002).fit(states)

    energies = transitioner.transform(
        states,
        state_labels=state_labels,
        order="combinations",
    )

    assert isinstance(energies.index, pd.MultiIndex)
    assert energies.index.names == ["condition", "state"]
    assert energies.index[0] == (
        ("baseline", "active"),
        ("rest", "task"),
    )


def test_transitioner_validates_fit_time_node_label_length(transition_data):
    """Check that node labels supplied to fit match the fitted node count."""
    adjacency, states = transition_data

    with pytest.raises(ValueError, match="node_labels must contain 2"):
        Transitioner(A=adjacency, T=0.002).fit(
            states,
            node_labels=["only-one"],
        )


def test_transitioner_validates_transform_time_state_label_length(
    transition_data,
):
    """Check that state labels supplied to transform match that transform input."""
    adjacency, states = transition_data
    transitioner = Transitioner(A=adjacency, T=0.002).fit(states)

    with pytest.raises(ValueError, match="state_labels must contain 3"):
        transitioner.transform(
            states,
            state_labels=["rest", "task"],
        )


def test_transitioner_transform_requires_fitted_input_type(transition_data):
    """Check that transform uses the same tabular/image modality as fit."""
    adjacency, states = transition_data
    transitioner = Transitioner(A=adjacency, T=0.002).fit(states)
    image = nib.Nifti1Image(
        states.T.reshape(2, 1, 1, 3),
        np.eye(4),
    )

    with pytest.raises(TypeError, match="must match the type used during fit"):
        transitioner.transform(image)


def test_transitioner_transform_requires_fitted_node_count(transition_data):
    """Check that transform data has the same number of nodes as fit data."""
    adjacency, states = transition_data
    transitioner = Transitioner(A=adjacency, T=0.002).fit(states)

    with pytest.raises(ValueError, match="same number of nodes as seen during fit"):
        transitioner.transform(np.ones((3, 3)))


def test_transitioner_rejects_pairwise_order_for_one_state(transition_data):
    """Check that transform rejects pairwise orders when only one state is given."""
    adjacency, states = transition_data
    transitioner = Transitioner(A=adjacency, T=0.002).fit(states)

    for order in ("combinations", "permutations"):
        with pytest.raises(ValueError, match="requires at least two states"):
            transitioner.transform(states[:1], order=order)


@pytest.mark.parametrize("order", ["product", "stability"])
def test_transitioner_allows_self_transition_for_one_state(
    transition_data, order
):
    """Check that product and stability can transform a single state."""
    adjacency, states = transition_data
    transitioner = Transitioner(A=adjacency, T=0.002).fit(states)

    energies = transitioner.transform(states[:1], order=order)

    assert energies.shape == (1, 2)


def test_transitioner_accepts_separate_source_and_target_states(
    transition_data,
):
    """Check that x0/xf input gives the same result as an equivalent state matrix."""
    adjacency, states = transition_data

    separate = Transitioner(A=adjacency, T=0.002).fit_transform(
        x0=states[0],
        xf=states[1],
        order="permutations",
    )
    matrix = Transitioner(A=adjacency, T=0.002).fit_transform(
        X=states[:2],
        order="permutations",
    )

    np.testing.assert_allclose(separate.to_numpy(), matrix.to_numpy())


@pytest.mark.parametrize(
    ("order", "n_transitions"),
    [
        ("combinations", 1),
        ("permutations", 2),
        ("product", 4),
        ("stability", 2),
    ],
)
def test_separate_states_honor_transition_order(
    transition_data, order, n_transitions
):
    """Check that x0/xf endpoints honor each supported transition-order mode."""
    adjacency, states = transition_data
    transitioner = Transitioner(A=adjacency, T=0.002)

    energies = transitioner.fit_transform(
        x0=states[0],
        xf=states[1],
        order=order,
    )

    assert energies.shape == (n_transitions, 2)


@pytest.mark.parametrize(
    ("parameter", "value"),
    [
        ("store_state_trajectories", None),
        ("store_state_trajectories", 1),
        ("store_control_trajectories", "yes"),
    ],
)
def test_transitioner_requires_boolean_storage_options(
    transition_data, parameter, value
):
    """Check that trajectory-storage settings accept only booleans."""
    adjacency, states = transition_data

    with pytest.raises(TypeError, match=f"{parameter} must be a boolean"):
        Transitioner(
            A=adjacency,
            T=0.002,
            **{parameter: value},
        ).fit(states)


def test_transitioner_can_disable_large_array_storage(transition_data):
    """Check that trajectory arrays are not retained when storage is disabled."""
    adjacency, states = transition_data
    transitioner = Transitioner(
        A=adjacency,
        T=0.002,
        store_state_trajectories=False,
        store_control_trajectories=False,
    )

    transitioner.fit_transform(states, order="combinations")
    trajectories, controls = transitioner.get_transition_arrays()

    assert trajectories is None
    assert controls is None


def test_transitioner_can_store_large_transition_arrays(transition_data):
    """Check that requested state and control trajectories are retained."""
    adjacency, states = transition_data
    transitioner = Transitioner(
        A=adjacency,
        T=0.002,
        store_state_trajectories=True,
        store_control_trajectories=True,
    )

    transitioner.fit_transform(states, order="combinations")
    trajectories, controls = transitioner.get_transition_arrays()

    assert trajectories.shape == (3, 2, 3)
    assert controls.shape == (3, 2, 3)


def test_transitioner_get_errors_requires_transform(transition_data):
    """Check that numerical errors are unavailable before transform is called."""
    adjacency, states = transition_data
    transitioner = Transitioner(A=adjacency, T=0.002).fit(states)

    with pytest.raises(Exception):
        transitioner.get_errors()


def test_transitioner_feature_names_use_fitted_node_labels(transition_data):
    """Check that sklearn feature names reflect the fitted node labels."""
    adjacency, states = transition_data
    node_labels = pd.Index(["A", "B"], name="node")
    transitioner = Transitioner(A=adjacency, T=0.002).fit(
        states,
        node_labels=node_labels,
    )

    assert transitioner.get_feature_names_out().tolist() == ["A", "B"]


def test_transitioner_caches_transition_computations(transition_data, tmp_path):
    """Check that identical transforms reuse cached trajectory computations."""
    adjacency, states = transition_data
    transitioner = Transitioner(
        A=adjacency,
        T=0.002,
        memory=tmp_path,
    ).fit(states)

    first = transitioner.transform(states, order="combinations")
    first_cache_entries = set(tmp_path.rglob("output.pkl"))

    second = transitioner.transform(states.copy(), order="combinations")

    assert first_cache_entries
    assert set(tmp_path.rglob("output.pkl")) == first_cache_entries
    pd.testing.assert_frame_equal(second, first)

    changed_states = states.copy()
    changed_states[0, 0] += 0.1
    transitioner.transform(changed_states, order="combinations")

    assert len(set(tmp_path.rglob("output.pkl"))) > len(first_cache_entries)


def test_transitioner_masks_a_4d_nifti_image(transition_data):
    """Check that a fitted labels masker maps image states into node states."""
    adjacency, states = transition_data
    labels = nib.Nifti1Image(
        np.array([1, 2], dtype=np.int16).reshape(2, 1, 1),
        np.eye(4),
    )
    image = nib.Nifti1Image(
        states.T.reshape(2, 1, 1, 3),
        np.eye(4),
    )
    masker = NiftiLabelsMasker(
        labels_img=labels,
        standardize=None,
        reports=False,
        keep_masked_labels=False,
    )

    image_energies = Transitioner(
        A=adjacency,
        T=0.002,
        masker=masker,
    ).fit_transform(
        image,
        order="combinations",
    )

    array_energies = Transitioner(
        A=adjacency,
        T=0.002,
    ).fit_transform(
        states,
        order="combinations",
    )

    assert image_energies.shape == (3, 2)
    np.testing.assert_allclose(
        image_energies.to_numpy(),
        array_energies.to_numpy(),
    )


def test_image_input_requires_a_masker(transition_data):
    """Check that image-like state input cannot be fitted without a masker."""
    adjacency, _ = transition_data
    image = nib.Nifti1Image(
        np.ones((2, 1, 1, 2)),
        np.eye(4),
    )

    with pytest.raises(ValueError, match="requires a masker"):
        Transitioner(A=adjacency, T=0.002).fit(image)


def test_masker_is_cloned_before_fitting(transition_data):
    """Check that fitting Transitioner does not fit the user-owned masker object."""
    adjacency, states = transition_data
    labels = nib.Nifti1Image(
        np.array([1, 2], dtype=np.int16).reshape(2, 1, 1),
        np.eye(4),
    )
    image = nib.Nifti1Image(
        states.T.reshape(2, 1, 1, 3),
        np.eye(4),
    )
    masker = NiftiLabelsMasker(
        labels_img=labels,
        reports=False,
        keep_masked_labels=False,
    )

    transitioner = Transitioner(
        A=adjacency,
        T=0.002,
        masker=masker,
    ).fit(image)

    assert transitioner.masker_ is not masker
    assert transitioner.masker_.n_elements_ == 2
