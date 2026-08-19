import inspect

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
    _set_transition_order,
    _validate_transition_inputs,
    get_state_to_state_df,
    state_to_state_integration,
    state_to_state_transition,
)

# TODO: Add little docstrings to each test function so we know what it is testing

@pytest.fixture
def transition_data():
    adjacency = np.array([[-1.0, 0.1], [0.1, -1.2]])
    states = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.25]])
    return adjacency, states


def _compute_validated_transition(
    A,
    T,
    B,
    X,
    rho=1,
    S="identity",
    energy_type="optimal",
    order="permutations",
    system="continuous",
):
    """Validate test inputs before calling the low-level compute function."""
    A, T, B, X, rho, S, _ = _validate_transition_inputs(
        A, T, B, X, rho, S, energy_type, order, system
    )
    return state_to_state_transition(
        A=A,
        T=T,
        B=B,
        X=X,
        rho=rho,
        S=S,
        energy_type=energy_type,
        order=order,
        system=system,
    )


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
    count, indices = _set_transition_order(3, order)
    assert count == len(expected)
    assert indices == expected


def test_set_transition_order_rejects_invalid_order():
    with pytest.raises(ValueError, match="order must be one of"):
        _set_transition_order(3, "invalid")


def test_transition_matches_nctpy(transition_data):
    adjacency, states = transition_data
    horizon = 0.002
    trajectories, control_trajectories, errors = _compute_validated_transition(
        A=adjacency,
        T=horizon,
        B="identity",
        X=states,
        order="permutations",
        system="continuous",
    )
    expected_x, expected_u, expected_error = get_control_inputs(
        A_norm=adjacency,
        T=horizon,
        B=np.eye(2),
        x0=states[0],
        xf=states[1],
        system="continuous",
    )

    assert trajectories.shape == (3, 2, 6)
    assert control_trajectories.shape == (3, 2, 6)
    assert errors.shape == (6, 2)
    np.testing.assert_allclose(trajectories[:, :, 0], expected_x)
    np.testing.assert_allclose(control_trajectories[:, :, 0], expected_u)
    np.testing.assert_allclose(errors[0], expected_error)

    energies = state_to_state_integration(control_trajectories)
    assert energies.shape == (6, 2)
    np.testing.assert_allclose(energies[0], integrate_u(expected_u))

def test_validate_transition_inputs_rejects_invalid_order(transition_data):
    adjacency, states = transition_data
    with pytest.raises(ValueError, match="order must be one of"):
        _validate_transition_inputs(
            adjacency,
            0.002,
            "identity",
            states,
            1,
            "identity",
            "optimal",
            "invalid",
            "continuous",
        )


def test_minimal_energy_matches_zero_trajectory_penalty(transition_data):
    adjacency, states = transition_data
    minimum = _compute_validated_transition(
        adjacency,
        0.002,
        "identity",
        states,
        rho=None,
        S=None,
        energy_type="minimal",
        order="combinations",
    )
    explicit = _compute_validated_transition(
        adjacency,
        0.002,
        "identity",
        states,
        rho=1,
        S=np.zeros((2, 2)),
        order="combinations",
    )
    for actual, expected in zip(minimum, explicit):
        np.testing.assert_allclose(actual, expected)


@pytest.mark.parametrize(
    ("energy_type", "rho", "S", "message"),
    [
        ("invalid", 1, "identity", "energy_type must be one of"),
        ("minimal", 1, None, "rho and S must both be None"),
        ("minimal", None, "identity", "rho and S must both be None"),
        ("optimal", None, None, "rho and S must both be provided"),
        ("optimal", 1, None, "rho and S must both be provided"),
    ],
)
def test_energy_type_validates_rho_and_S(
    transition_data, energy_type, rho, S, message
):
    adjacency, states = transition_data
    with pytest.raises(ValueError, match=message):
        _validate_transition_inputs(
            adjacency,
            0.002,
            "identity",
            states,
            rho,
            S,
            energy_type,
            "permutations",
            "continuous",
        )


def test_transitioner_rejects_minimal_energy_with_default_rho_and_S(
    transition_data,
):
    adjacency, states = transition_data
    transitioner = Transitioner(A=adjacency, T=0.002, energy_type="minimal")
    with pytest.raises(ValueError, match="rho and S must both be None"):
        transitioner.fit(states)


def test_transitioner_accepts_minimal_energy_with_rho_and_S_none(
    transition_data,
):
    adjacency, states = transition_data
    transitioner = Transitioner(
        A=adjacency,
        T=0.002,
        energy_type="minimal",
        rho=None,
        S=None,
        order="combinations",
    )
    energies = transitioner.fit_transform(states)
    assert energies.shape == (3, 2)

def test_discrete_transition_preserves_integer_horizon():
    adjacency = np.diag([0.3, 0.2])
    states = np.array([[1.0, 0.0], [0.0, 1.0]])
    trajectories, control_trajectories, errors = _compute_validated_transition(
        adjacency,
        2,
        "identity",
        states,
        order="combinations",
        system="discrete",
    )
    assert trajectories.shape == (3, 2, 1)
    assert control_trajectories.shape == (2, 2, 1)
    assert errors.shape == (1, 2)

def test_state_to_state_dataframe_has_only_label_parameters():
    assert list(inspect.signature(get_state_to_state_df).parameters) == [
        "state_to_state_array",
        "order",
        "node_labels",
        "state_labels",
    ]

def test_transitioner_is_the_only_public_transition_class():
    assert transitions.__all__[0] == "Transitioner"
    assert not hasattr(transitions, "TransitionTransformer")
    assert not hasattr(transitions, "StateTransitionTransformer")

def test_removed_transition_helpers_are_not_public():
    assert not hasattr(transitions, "state_to_state_aggregation")
    assert not hasattr(transitions, "state_to_state_comparison")
    assert not hasattr(transitions, "get_state_comparison_df")

def test_state_labels_accept_a_named_index():
    state_labels = pd.Index(["rest", "task"], name="condition")
    result = get_state_to_state_df(
        np.ones((2, 1)),
        "stability",
        state_labels=state_labels,
    )
    assert result.index.names == ["endpoint", "condition"]
    assert result.index.tolist() == [
        (("source", "target"), ("rest", "rest")),
        (("source", "target"), ("task", "task")),
    ]

def test_state_labels_accept_multiindex_like_tuples():
    state_labels = [
        ("baseline", "rest"),
        ("active", "task"),
        ("baseline", "recovery"),
    ]
    result = get_state_to_state_df(
        np.ones((3, 1)),
        "combinations",
        state_labels=state_labels,
    )
    assert result.index.names == [
        "endpoint",
        "state_label_0",
        "state_label_1",
    ]
    assert result.index[0] == (
        ("source", "target"),
        ("baseline", "active"),
        ("rest", "task"),
    )

def test_state_labels_validate_input_and_transition_count():
    with pytest.raises(ValueError, match="rows do not match state_labels"):
        get_state_to_state_df(
            np.ones((2, 1)),
            "combinations",
            state_labels=["rest", "task", "recovery"],
        )
    with pytest.raises(TypeError, match="list-like or MultiIndex-like"):
        get_state_to_state_df(
            np.ones((1, 1)),
            "stability",
            state_labels={"state": ["rest"]},
        )

def test_state_to_state_dataframe_with_hierarchical_labels():
    energies = np.arange(12, dtype=float).reshape(6, 2)
    node_labels = pd.MultiIndex.from_arrays(
        [["hemisphere", "hemisphere"], ["left", "right"]],
        names=["node_group", "node_name"],
    )
    state_labels = pd.MultiIndex.from_arrays(
        [
            ["baseline", "active", "baseline"],
            ["rest", "task", "recovery"],
        ],
        names=["group", "state"],
    )
    kwargs = {
        "node_labels": node_labels,
        "state_labels": state_labels,
    }

    result = get_state_to_state_df(energies, "permutations", **kwargs)
    assert result.shape == (6, 2)
    assert result.index.names == [
        "endpoint",
        "group",
        "state",
    ]
    assert result.columns.equals(node_labels)
    np.testing.assert_array_equal(result.to_numpy(), energies)

def test_state_to_state_dataframe_accepts_a_multiindex():
    energies = np.arange(12, dtype=float).reshape(3, 4)
    node_labels = pd.MultiIndex.from_arrays(
        [
            ["association", "association", "sensory", "sensory"],
            ["default", "default", "visual", "visual"],
            ["medial", "lateral", "dorsal", "ventral"],
            ["A", "B", "C", "D"],
        ],
        names=["cortex", "network", "region", "node"],
    )

    result = get_state_to_state_df(
        energies,
        "combinations",
        node_labels=node_labels,
        state_labels=["rest", "task", "recovery"],
    )
    assert result.shape == (3, 4)
    assert result.columns.equals(node_labels)
    assert result.columns[2] == ("sensory", "visual", "dorsal", "C")

def test_node_labels_accept_a_list_like_object():
    energies = np.arange(4, dtype=float).reshape(2, 2)
    node_labels = pd.Index(
        ["shared-label", "shared-label"], name="node"
    )

    result = get_state_to_state_df(
        energies, "stability", node_labels=node_labels
    )
    assert result.columns.equals(node_labels)
    np.testing.assert_array_equal(result.to_numpy(), energies)

def test_node_labels_accept_multiindex_like_tuples():
    energies = np.arange(6, dtype=float).reshape(2, 3)
    node_labels = [
        ("default", "A"),
        ("default", "B"),
        ("visual", "C"),
    ]

    result = get_state_to_state_df(
        energies, "stability", node_labels=node_labels
    )
    assert isinstance(result.columns, pd.MultiIndex)
    assert result.columns.tolist() == node_labels


def test_plain_list_creates_a_regular_column_index():
    result = get_state_to_state_df(
        np.ones((1, 2)),
        "stability",
        node_labels=["A", "B"],
    )
    assert isinstance(result.columns, pd.Index)
    assert not isinstance(result.columns, pd.MultiIndex)
    assert result.columns.tolist() == ["A", "B"]

def test_node_labels_validate_input_and_length():
    energies = np.ones((1, 2))
    with pytest.raises(ValueError, match="must contain 2 values"):
        get_state_to_state_df(
            energies, "stability", node_labels=["A"]
        )
    with pytest.raises(ValueError, match="tuples of equal length"):
        get_state_to_state_df(
            energies,
            "stability",
            node_labels=[("network", "A"), ("network", "visual", "B")],
        )
    with pytest.raises(TypeError, match="list-like or MultiIndex-like"):
        get_state_to_state_df(
            energies, "stability", node_labels="not-list-like"
        )
    with pytest.raises(TypeError, match="list-like or MultiIndex-like"):
        get_state_to_state_df(
            energies, "stability", node_labels={"node": ["A", "B"]}
        )
    with pytest.raises(TypeError, match="list-like or MultiIndex-like"):
        get_state_to_state_df(
            energies,
            "stability",
            node_labels=pd.DataFrame({"node": ["A", "B"]}),
        )


def test_transitioner_with_state_matrix(transition_data):
    adjacency, states = transition_data
    transitioner = Transitioner(
        A=adjacency, T=0.002, order="combinations"
    )
    assert clone(transitioner).get_params()["order"] == "combinations"

    energies = transitioner.fit_transform(states)
    assert energies.shape == (3, 2)
    assert transitioner.transition_indices_ == [(0, 1), (0, 2), (1, 2)]
    assert transitioner.get_errors().shape == (3, 2)
    trajectories, control_trajectories = transitioner.get_transition_arrays()
    assert trajectories.shape == control_trajectories.shape == (3, 2, 3)
    assert transitioner.get_feature_names_out().tolist() == ["node_0", "node_1"]


def test_transitioner_returns_labelled_dataframe(transition_data):
    adjacency, states = transition_data
    node_labels = pd.MultiIndex.from_tuples(
        [("left", "A"), ("right", "B")], names=["hemisphere", "node"]
    )
    transitioner = Transitioner(
        A=adjacency,
        T=0.002,
        order="combinations",
        node_labels=node_labels,
        state_labels=["rest", "task", "recovery"],
    )

    energies = transitioner.fit_transform(states)

    assert isinstance(energies, pd.DataFrame)
    assert energies.columns.equals(node_labels)
    assert energies.index.names == ["endpoint", "state"]
    assert transitioner.get_feature_names_out().tolist() == list(
        node_labels
    )


def test_transitioner_infers_labels_from_dataframe(transition_data):
    adjacency, states = transition_data
    node_labels = pd.MultiIndex.from_tuples(
        [("left", "A"), ("right", "B")], names=["hemisphere", "node"]
    )
    state_labels = pd.MultiIndex.from_tuples(
        [
            ("baseline", "rest"),
            ("active", "task"),
            ("baseline", "recovery"),
        ],
        names=["condition", "state"],
    )
    states_df = pd.DataFrame(
        states,
        index=state_labels,
        columns=node_labels,
    )

    inferred = Transitioner(
        A=adjacency,
        T=0.002,
        order="combinations",
    ).fit_transform(X=states_df)
    explicit = Transitioner(
        A=adjacency,
        T=0.002,
        order="combinations",
        node_labels=node_labels,
        state_labels=state_labels,
    ).fit_transform(X=states)

    assert isinstance(inferred, pd.DataFrame)
    assert inferred.columns.equals(node_labels)
    assert inferred.index.names == [
        "endpoint",
        "condition",
        "state",
    ]
    pd.testing.assert_frame_equal(inferred, explicit)


def test_transitioner_validates_label_lengths(transition_data):
    adjacency, states = transition_data
    with pytest.raises(ValueError, match="node_labels must contain 2"):
        Transitioner(
            A=adjacency, T=0.002, node_labels=["only-one"]
        ).fit(states)
    with pytest.raises(ValueError, match="state_labels must contain 3"):
        Transitioner(
            A=adjacency, T=0.002, state_labels=["rest", "task"]
        ).fit(states)


@pytest.mark.parametrize(
    ("parameter", "value"),
    [
        ("store_trajectories", None),
        ("store_trajectories", 1),
        ("store_control_trajectories", "yes"),
    ],
)
def test_transitioner_requires_boolean_storage_options(
    transition_data, parameter, value
):
    adjacency, states = transition_data
    with pytest.raises(TypeError, match=f"{parameter} must be a boolean"):
        Transitioner(
            A=adjacency,
            T=0.002,
            **{parameter: value},
        ).fit(states)


def test_transitioner_can_disable_large_array_storage(transition_data):
    adjacency, states = transition_data
    transitioner = Transitioner(
        A=adjacency,
        T=0.002,
        order="combinations",
        store_trajectories=False,
        store_control_trajectories=False,
    )

    energies = transitioner.fit_transform(states)
    trajectories, control_trajectories = transitioner.get_transition_arrays()

    assert energies.shape == (3, 2)
    assert trajectories is None
    assert control_trajectories is None
    assert not hasattr(transitioner, "trajectories_")
    assert not hasattr(transitioner, "control_trajectories_")


def test_transitioner_can_use_pre_normalized_adjacency(transition_data):
    adjacency, states = transition_data
    transitioner = Transitioner(
        A=adjacency, T=0.002, normalize_A=False
    ).fit(X=states)

    assert clone(transitioner).get_params()["normalize_A"] is False
    np.testing.assert_array_equal(transitioner.A_, adjacency)


def test_transitioner_normalizes_adjacency_by_default(transition_data):
    adjacency, states = transition_data
    transitioner = Transitioner(A=adjacency, T=0.002).fit(X=states)

    parameters = clone(transitioner).get_params()
    assert parameters["normalize_A"] is True
    assert parameters["c"] == 1
    expected = matrix_normalization(adjacency, system="continuous", c=1)
    np.testing.assert_allclose(transitioner.A_, expected)
    assert transitioner.c_ == 1.0
    assert isinstance(transitioner.c_, float)


@pytest.mark.parametrize("system", ["continuous", "discrete"])
def test_transitioner_can_normalize_adjacency(transition_data, system):
    _, states = transition_data
    adjacency = np.array([[0.0, 2.0], [1.0, 0.0]])
    horizon = 0.002 if system == "continuous" else 2
    transitioner = Transitioner(
        A=adjacency,
        T=horizon,
        system=system,
        normalize_A=True,
        c=2,
    ).fit(X=states)

    expected = matrix_normalization(adjacency, system=system, c=2)
    np.testing.assert_allclose(transitioner.A_, expected)
    np.testing.assert_array_equal(adjacency, [[0.0, 2.0], [1.0, 0.0]])


@pytest.mark.parametrize("normalize_A", [None, 1, "yes"])
def test_transitioner_requires_boolean_normalize_A(
    transition_data, normalize_A
):
    adjacency, states = transition_data
    with pytest.raises(TypeError, match="normalize_A must be a boolean"):
        Transitioner(
            A=adjacency, T=0.002, normalize_A=normalize_A
        ).fit(X=states)


@pytest.mark.parametrize("c", [None, True, 0, -1, np.inf, "1"])
def test_transitioner_requires_positive_finite_c(transition_data, c):
    adjacency, states = transition_data
    with pytest.raises(ValueError, match="c must be a positive finite number"):
        Transitioner(A=adjacency, T=0.002, c=c).fit(X=states)


def test_transitioner_accepts_separate_source_and_target_states(transition_data):
    adjacency, states = transition_data
    separate = Transitioner(
        A=adjacency, T=0.002, order="permutations"
    ).fit_transform(x0=states[0], xf=states[1])
    matrix = Transitioner(
        A=adjacency, T=0.002, order="permutations"
    ).fit_transform(X=states[:2])

    assert separate.shape == (2, 2)
    np.testing.assert_allclose(separate, matrix)


def test_transitioner_can_transform_a_separate_source_and_target(
    transition_data,
):
    adjacency, states = transition_data
    transitioner = Transitioner(A=adjacency, T=0.002).fit(X=states)

    energies = transitioner.transform(x0=states[1], xf=states[2])

    assert energies.shape == (2, 2)
    assert transitioner.transition_indices_ == [(0, 1), (1, 0)]


@pytest.mark.parametrize(
    ("order", "expected_indices"),
    [
        ("combinations", [(0, 1)]),
        ("permutations", [(0, 1), (1, 0)]),
        ("product", [(0, 0), (0, 1), (1, 0), (1, 1)]),
        ("stability", [(0, 0), (1, 1)]),
    ],
)
def test_separate_states_honor_transition_order(
    transition_data, order, expected_indices
):
    adjacency, states = transition_data
    transitioner = Transitioner(A=adjacency, T=0.002, order=order)

    energies = transitioner.fit_transform(x0=states[0], xf=states[1])

    assert energies.shape == (len(expected_indices), 2)
    assert transitioner.transition_indices_ == expected_indices


def test_transitioner_requires_one_complete_state_input_mode(transition_data):
    adjacency, states = transition_data
    transitioner = Transitioner(A=adjacency, T=0.002)

    with pytest.raises(ValueError, match="Provide either X or both x0 and xf"):
        transitioner.fit()
    with pytest.raises(ValueError, match="x0 and xf must be provided together"):
        transitioner.fit(x0=states[0])
    with pytest.raises(ValueError, match="Provide either X or x0 and xf, not both"):
        transitioner.fit(X=states, x0=states[0], xf=states[1])


def test_transitioner_validates_separate_state_shapes(transition_data):
    adjacency, states = transition_data
    transitioner = Transitioner(A=adjacency, T=0.002)

    with pytest.raises(ValueError, match="x0 must be a one-dimensional state"):
        transitioner.fit(x0=states[[0]], xf=states[1])
    with pytest.raises(ValueError, match="same number of nodes"):
        transitioner.fit(x0=states[0], xf=np.ones(3))


def test_transitioner_inherits_nilearn_cache_mixin():
    assert issubclass(Transitioner, CacheMixin)


def test_transitioner_caches_transition_computations(transition_data, tmp_path):
    adjacency, states = transition_data
    transitioner = Transitioner(
        A=adjacency,
        T=0.002,
        order="combinations",
        memory=tmp_path,
    )

    first = transitioner.fit_transform(states)
    first_cache_entries = set(tmp_path.rglob("output.pkl"))
    assert len(first_cache_entries) == 1

    second = transitioner.transform(states.copy())
    assert set(tmp_path.rglob("output.pkl")) == first_cache_entries
    np.testing.assert_allclose(second, first)

    changed_states = states.copy()
    changed_states[0, 0] += 0.1
    transitioner.transform(changed_states)
    assert len(set(tmp_path.rglob("output.pkl"))) == 2


def test_transitioner_masks_a_4d_nifti_image(transition_data):
    adjacency, states = transition_data
    labels = nib.Nifti1Image(
        np.array([1, 2], dtype=np.int16).reshape(2, 1, 1), np.eye(4)
    )
    image = nib.Nifti1Image(states.T.reshape(2, 1, 1, 3), np.eye(4))
    masker = NiftiLabelsMasker(
        labels_img=labels,
        standardize=None,
        reports=False,
        keep_masked_labels=False,
    )
    transitioner = Transitioner(
        A=adjacency,
        T=0.002,
        order="combinations",
        masker=masker,
    )

    image_energies = transitioner.fit_transform(image)
    array_energies = Transitioner(
        A=adjacency, T=0.002, order="combinations"
    ).fit_transform(states)

    assert image_energies.shape == (3, 2)
    np.testing.assert_allclose(image_energies, array_energies)


def test_image_input_requires_a_masker(transition_data):
    adjacency, _ = transition_data
    image = nib.Nifti1Image(np.ones((2, 1, 1, 2)), np.eye(4))
    with pytest.raises(ValueError, match="requires a masker"):
        Transitioner(A=adjacency, T=0.002).fit(image)
