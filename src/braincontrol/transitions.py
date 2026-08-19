"""State-to-state transition utilities.

This module is the successor of :mod:`nict.single_subject`.  The functional
API accepts state matrices directly, while :class:`Transitioner`
also accepts image-like inputs through a scikit-learn compatible masker (for
example, :class:`nilearn.maskers.NiftiLabelsMasker`).
"""

from collections.abc import Mapping
from itertools import combinations, permutations, product

import numpy as np
import pandas as pd
from nctpy.energies import get_control_inputs, integrate_u
from nctpy.utils import matrix_normalization
from nilearn._utils.cache_mixin import CacheMixin
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.utils.validation import check_is_fitted

########################################################################################
## Input validation
########################################################################################

# FIXME: Add docs here what this function is doing.
# FIXME: We don't have to work with try-except? The input is either
# X, or [x0,xf], or 4D-iimg or [3DImg,3dimg] so we can use is-Niimg-like?
# https://nilearn.github.io/dev/modules/generated/nilearn.image.check_niimg.html#nilearn.image.check_niimg
def _is_state_matrix(X):
    try:
        array = np.asarray(X)
    except Exception:
        return False
    return array.ndim == 2 and np.issubdtype(array.dtype, np.number)

def _as_2d_float_array(value, name):
    """Check that input is (or can be converted into) a finite, two-dimensional floating-point array."""
    array = np.asarray(value, dtype=float)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional array")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array

def _resolve_square_matrix(value, name, n_nodes):
    """Return a square matrix with one row and column per node."""
    if isinstance(value, str):
        if value != "identity":
            raise ValueError(f"{name} must be an array or 'identity'")
        return np.eye(n_nodes)

    matrix = _as_2d_float_array(value, name)
    if matrix.shape != (n_nodes, n_nodes):
        raise ValueError(
            f"{name} must have shape {(n_nodes, n_nodes)}; got {matrix.shape}"
        )
    return matrix

def _validate_choice(value, name, choices):
    """Validate an enumerated option and return it unchanged."""
    if value not in choices:
        formatted_choices = ", ".join(repr(choice) for choice in choices)
        raise ValueError(
            f"{name} must be one of {formatted_choices}; got {value!r}"
        )
    return value

def _validate_boolean(value, name):
    """Validate and return a boolean option."""
    if not isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a boolean")
    return bool(value)

def _validate_normalization_inputs(normalize_A, c):
    """Validate adjacency-normalization parameters."""
    normalize_A = _validate_boolean(normalize_A, "normalize_A")
    if (
        isinstance(c, (bool, np.bool_))
        or not isinstance(c, (int, float, np.integer, np.floating))
        or not np.isfinite(c)
        or c <= 0
    ):
        raise ValueError("c must be a positive finite number")
    return normalize_A, float(c)

def _resolve_state_input(X=None, x0=None, xf=None):
    """Validate and return one of the supported state-input forms. Returns
    either X or [x0,xf]"""
    endpoints_provided = x0 is not None or xf is not None
    if X is not None and endpoints_provided:
        raise ValueError("Provide either X or x0 and xf, not both")
    if X is not None:
        return X
    if x0 is None and xf is None:
        raise ValueError("Provide either X or both x0 and xf")
    if x0 is None or xf is None:
        raise ValueError("x0 and xf must be provided together")

    endpoint_arrays = []
    endpoint_is_numeric = []
    for endpoint, name in ((x0, "x0"), (xf, "xf")):
        try:
            array = np.asarray(endpoint)
        except Exception:
            array = None
        is_numeric = array is not None and np.issubdtype(array.dtype, np.number)
        endpoint_is_numeric.append(is_numeric)
        if is_numeric:
            if array.ndim != 1:
                raise ValueError(f"{name} must be a one-dimensional state")
            endpoint_arrays.append(array)

    if all(endpoint_is_numeric):
        if endpoint_arrays[0].shape != endpoint_arrays[1].shape:
            raise ValueError("x0 and xf must contain the same number of nodes")
        return np.stack(endpoint_arrays)
    if any(endpoint_is_numeric):
        raise TypeError("x0 and xf must use the same input type")
    return [x0, xf]

def _validate_transition_inputs(
    A,
    T,
    B,
    X,
    rho,
    S,
    energy_type,
    order,
    system,
    *,
    normalize_A=False,
    c=1,
):
    """Validate and normalize inputs used to compute control energy."""

    order = _validate_choice(
        order,
        "order",
        ("combinations", "permutations", "product", "stability"),
    )
    energy_type = _validate_choice(
        energy_type, "energy_type", ("minimal", "optimal")
    )
    system = _validate_choice(system, "system", ("continuous", "discrete"))
    normalize_A, c = _validate_normalization_inputs(normalize_A, c)

    A = _as_2d_float_array(A, "A")
    if A.shape[0] != A.shape[1]:
        raise ValueError(f"A must be square; got shape {A.shape}")
    if normalize_A:
        A = matrix_normalization(A, system=system, c=c)

    X = _as_2d_float_array(X, "X")
    n_nodes = A.shape[0]
    if X.shape[1] != n_nodes:
        raise ValueError(
            "X must have one column per node in A; "
            f"got {X.shape[1]} columns for {n_nodes} nodes"
        )

    if not np.isscalar(T) or not np.isfinite(T) or T <= 0:
        raise ValueError("T must be a positive number")
    if system == "discrete" and (int(T) != T or T < 2):
        raise ValueError("T must be an integer of at least 2 for a discrete system")
    B = _resolve_square_matrix(B, "B", n_nodes)
    if energy_type == "minimal":
        if rho is not None or S is not None:
            raise ValueError(
                "rho and S must both be None when energy_type='minimal'"
            )
    else:
        if rho is None or S is None:
            raise ValueError(
                "rho and S must both be provided when energy_type='optimal'"
            )
        if not np.isscalar(rho) or not np.isfinite(rho) or rho <= 0:
            raise ValueError("rho must be a positive number")
        S = _resolve_square_matrix(S, "S", n_nodes)
    T = int(T) if system == "discrete" else float(T)
    rho = None if rho is None else float(rho)
    return A, T, B, X, rho, S, c

########################################################################################
## Transitions
########################################################################################

def _set_transition_order(n_states, order):
    """Return the number and indices of requested state transitions.

    Each transition tuple contains ``(transition, source, target)``.  The
    ordering matches the corresponding iterator in :mod:`itertools`.
    """
    if not isinstance(n_states, (int, np.integer)) or isinstance(n_states, bool):
        raise TypeError("n_states must be an integer")
    if n_states < 1:
        raise ValueError("n_states must be at least 1")
    order = _validate_choice(
        order,
        "order",
        ("combinations", "permutations", "product", "stability"),
    )

    indices = range(n_states)
    if order == "permutations":
        pairs = permutations(indices, r=2)
    elif order == "combinations":
        pairs = combinations(indices, r=2)
    elif order == "product":
        pairs = product(indices, repeat=2)
    else:
        pairs = ((index, index) for index in indices)

    transition_indices = [
        (transition, source, target)
        for transition, (source, target) in enumerate(pairs)
    ]
    return len(transition_indices), transition_indices


# FIXME: Should be renamed to state_to_state_trajectories
# FIXME: trajectories should be always named state_trajectories (throughout the whole script). That means we
# have state_trajectories and control_trajectories.
def state_to_state_transition(
    A,
    T,
    B,
    X,
    rho,
    S,
    energy_type,
    order,
    *,
    system="continuous",
    xr="zero",
    expm_version="scipy",
):
    """Compute state and control trajectories from prevalidated inputs.

    Parameters
    ----------
    A : ndarray of shape (n_nodes, n_nodes)
        A validated, normalized adjacency matrix.
    T : float
        Time horizon.
    B : ndarray of shape (n_nodes, n_nodes)
        Validated control input matrix.
    X : ndarray of shape (n_states, n_nodes)
        Validated state matrix with one state per row.
    rho : float or None
        Mixing parameter used by :func:`nctpy.energies.get_control_inputs`.
        Must be ``None`` for minimal energy and positive for optimal energy.
    S : ndarray of shape (n_nodes, n_nodes) or None
        Validated state-trajectory constraint matrix. Must be ``None`` for
        minimal energy and provided for optimal energy.
    energy_type : {"minimal", "optimal"}
        Type of control energy to compute. ``"minimal"`` requires ``rho`` and
        ``S`` to both be ``None``. ``"optimal"`` requires both parameters.
    order : {"combinations", "permutations", "product", "stability"}
        State-pair selection and ordering.
    system : {"continuous", "discrete"}, default="continuous"
        Time system used to normalize ``A``.
    xr : array-like or str, default="zero"
        Reference state passed to ``nctpy``.
    expm_version : {"scipy", "eig"}, default="scipy"
        Matrix-exponential implementation used by ``nctpy``.

    Returns
    -------
    trajectories : ndarray of shape (n_time_points, n_nodes, n_transitions)
    control_trajectories : ndarray of shape \
            (n_control_points, n_nodes, n_transitions)
    errors : ndarray of shape (n_transitions, 2)
        Numerical errors reported by ``nctpy`` for each transition.
    """
    n_transitions, transition_indices = _set_transition_order(X.shape[0], order)

    if energy_type == "minimal":
        solver_rho = 1
        solver_S = np.zeros_like(A)
    else:
        solver_rho = rho
        solver_S = S

    trajectories = []
    control_trajectories = []
    errors = []
    for _, source, target in transition_indices:
        trajectory, control_trajectory, error = get_control_inputs(
            A_norm=A,
            T=T,
            B=B,
            x0=X[source],
            xf=X[target],
            system=system,
            rho=solver_rho,
            S=solver_S,
            xr=xr,
            expm_version=expm_version,
        )
        trajectories.append(trajectory)
        control_trajectories.append(control_trajectory)
        errors.append(error)
        
    # TODO: I don't get this part. Why would n_transitions ever be None?
    # NOTE: The absolute minium that we can do is to provide on state
    # and to compute stability.
    if n_transitions:
        trajectory_array = np.stack(trajectories, axis=2)
        control_trajectory_array = np.stack(control_trajectories, axis=2)
        error_array = np.asarray(errors, dtype=float)
    # TODO: I also don't get this part.
    else:
        if system == "continuous":
            n_trajectory_points = round(T / 0.001) + 1
            n_control_points = n_trajectory_points
        else:
            n_trajectory_points = int(T) + 1
            n_control_points = int(T)
        trajectory_array = np.empty((n_trajectory_points, A.shape[0], 0))
        control_trajectory_array = np.empty(
            (n_control_points, A.shape[0], 0)
        )
        error_array = np.empty((0, 2))

    return trajectory_array, control_trajectory_array, error_array

def state_to_state_integration(control_trajectories):
    """Integrate squared control trajectories for every transition."""
    
    # FIXME: This function should not validate input. If this has to be
    # validated it should be done somehwere else.
    control_trajectories = np.asarray(control_trajectories, dtype=float)
    if control_trajectories.ndim != 3:
        raise ValueError(
            "control_trajectories must have shape "
            "(n_time_points, n_nodes, n_transitions)"
        )
    if not np.all(np.isfinite(control_trajectories)):
        raise ValueError(
            "control_trajectories must contain only finite values"
        )

    # FIXME: we can keep the following
    n_nodes, n_transitions = control_trajectories.shape[1:]
    energies = np.empty((n_transitions, n_nodes))
    for transition in range(n_transitions):
        energies[transition] = integrate_u(
            control_trajectories[:, :, transition]
        )
    return energies

def _coerce_labels(labels, expected_length, parameter_name):
    """Return list-like labels as a pandas Index or MultiIndex."""
    if isinstance(labels, (str, bytes, Mapping, pd.DataFrame)):
        raise TypeError(
            f"{parameter_name} must be a list-like or MultiIndex-like object"
        )

    if isinstance(labels, pd.MultiIndex):
        index = labels.copy()
    else:
        try:
            values = list(labels)
        except TypeError as error:
            raise TypeError(
                f"{parameter_name} must be a list-like or MultiIndex-like object"
            ) from error

        is_multiindex_like = bool(values) and all(
            isinstance(value, tuple) and len(value) > 1 for value in values
        )
        if is_multiindex_like:
            tuple_lengths = {len(value) for value in values}
            if len(tuple_lengths) != 1:
                raise ValueError(
                    f"MultiIndex-like {parameter_name} must use tuples of equal length"
                )
            names = getattr(labels, "names", None)
            index = pd.MultiIndex.from_tuples(values, names=names)
        else:
            name = getattr(labels, "name", None)
            index = pd.Index(values, name=name)

    if expected_length is not None and len(index) != expected_length:
        raise ValueError(
            f"{parameter_name} must contain {expected_length} values; "
            f"got {len(index)}"
        )
    if len(index) == 0:
        raise ValueError(f"{parameter_name} must contain at least one value")
    return index

# TODO: I still don't get what this is doing.
def _label_level_names(labels, default_name, label_prefix=None):
    """Return unique, non-conflicting names for label levels."""
    if isinstance(labels, pd.MultiIndex):
        source_names = labels.names
    else:
        source_names = [labels.name]

    names = []
    label_prefix = label_prefix or default_name
    for level, source_name in enumerate(source_names):
        fallback = (
            default_name
            if len(source_names) == 1
            else f"{label_prefix}_label_{level}"
        )
        name = source_name
        if not isinstance(name, str) or not name or name in names:
            name = fallback
        names.append(name)
    return names

def _state_transition_index(state_labels, order, n_transitions):
    """Build an endpoint-aware row index from state labels."""
    state_labels = _coerce_labels(state_labels, None, "state_labels")
    expected_transitions, transition_indices = _set_transition_order(
        len(state_labels), order
    )
    if expected_transitions != n_transitions:
        raise ValueError(
            "state_to_state_array rows do not match state_labels and order"
        )

    level_names = _label_level_names(state_labels, "state")
    if isinstance(state_labels, pd.MultiIndex):
        level_values = [
            state_labels.get_level_values(level).to_numpy()
            for level in range(state_labels.nlevels)
        ]
    else:
        level_values = [state_labels.to_numpy()]

    source_arrays = [
        [values[source] for _, source, _ in transition_indices]
        for values in level_values
    ]
    target_arrays = [
        [values[target] for _, _, target in transition_indices]
        for values in level_values
    ]

    endpoint_values = [("source", "target")] * n_transitions
    paired_label_values = [
        list(zip(source_values, target_values))
        for source_values, target_values in zip(source_arrays, target_arrays)
    ]
    return pd.MultiIndex.from_arrays(
        [endpoint_values, *paired_label_values],
        names=["endpoint", *level_names],
    )

def get_state_to_state_df(
    state_to_state_array,
    order,
    node_labels=None,
    state_labels=None,
):
    """Create a labelled DataFrame from a transition-by-node array.

    Parameters
    ----------
    state_to_state_array : array-like of shape (n_transitions, n_nodes)
        Values produced for every transition and node.
    order : {"combinations", "permutations", "product", "stability"}
        Ordering used to construct the transitions.
    node_labels : list-like or MultiIndex-like, optional
        One label or tuple of hierarchical labels per node. A pandas
        ``MultiIndex`` is preserved directly. A list-like object containing
        equal-length tuples is converted to a ``MultiIndex``. Other list-like
        values become a regular ``Index``. This is the only parameter for node
        metadata.
    state_labels : list-like or MultiIndex-like, optional
        One label or tuple of hierarchical labels per state. The output row
        index has an outer ``endpoint`` level containing the ordered pair
        ``("source", "target")``. Each original state-label level stores its
        corresponding ``(source_value, target_value)`` pair.
    Returns
    -------
    pandas.DataFrame
        A transition-by-node table.
    """
    values = _as_2d_float_array(state_to_state_array, "state_to_state_array")
    n_transitions, n_nodes = values.shape
    if node_labels is None:
        node_labels = pd.RangeIndex(n_nodes)
    else:
        node_labels = _coerce_labels(node_labels, n_nodes, "node_labels")

    transition_index = None
    if state_labels is not None:
        transition_index = _state_transition_index(
            state_labels, order, n_transitions
        )
    return pd.DataFrame(values, index=transition_index, columns=node_labels)

class Transitioner(
    TransformerMixin, CacheMixin, BaseEstimator, auto_wrap_output_keys=None
):
    """Transform states or labelled NIfTI images into transition energies.

    ``X`` passed to :meth:`fit` and :meth:`transform` can be a two-dimensional
    ``(n_states, n_nodes)`` array.  For image-like input, pass a compatible
    masker such as ``NiftiLabelsMasker(labels_img=atlas)`` to the constructor;
    each volume in a 4D image is then treated as one state.

    The fitted transformer exposes retained unintegrated results through
    :meth:`get_transition_arrays` and numerical errors through
    :meth:`get_errors`.

    State input can be provided either as ``X``, containing one state per row,
    or as the separate ``x0`` and ``xf`` keyword arguments. In either form,
    ``order`` controls which directed and self transitions are computed.

    Parameters
    ----------
    A : array-like of shape (n_nodes, n_nodes)
        Adjacency matrix. It is normalized during :meth:`fit` when
        ``normalize_A=True``.
    T : float
        Positive time horizon. Discrete systems require an integer of at least
        two.
    B : array-like of shape (n_nodes, n_nodes) or ``"identity"``
        Control input matrix.
    rho : float or None, default=1
        Positive mixing parameter for optimal control energy.
    S : array-like, ``"identity"``, or None, default="identity"
        State-trajectory constraint matrix for optimal control energy.
    energy_type : {"minimal", "optimal"}, default="optimal"
        Type of control energy to compute. ``"minimal"`` is mutually exclusive
        with ``rho`` and ``S``, so both must be ``None``. ``"optimal"`` requires
        both parameters to be provided.
    order : {"combinations", "permutations", "product", "stability"}, \
            default="permutations"
        State-pair selection and ordering.
    system : {"continuous", "discrete"}, default="continuous"
        Time system used for adjacency normalization and control computation.
    xr : array-like or str, default="zero"
        Reference state forwarded to :func:`nctpy.energies.get_control_inputs`.
    expm_version : {"scipy", "eig"}, default="scipy"
        Matrix-exponential implementation forwarded to ``nctpy``.
    masker : transformer, optional
        Scikit-learn compatible masker used for image-like state inputs.
    normalize_A : bool, default=True
        If ``True``, normalize ``A`` during :meth:`fit` with
        :func:`nctpy.utils.matrix_normalization` for the selected ``system``.
        If ``False``, use ``A`` as provided.
    c : float, default=1
        Positive normalization constant passed to
        :func:`nctpy.utils.matrix_normalization`.
    memory : None, str, pathlib.Path, or joblib.Memory, default=None
        Cache location for state-transition computations. Caching is disabled
        when ``None``.
    memory_level : int, default=1
        Cache state-transition computations when this value is at least 1.
    verbose : int, default=0
        Verbosity forwarded to Nilearn's caching infrastructure.
    node_labels : list-like or MultiIndex-like, optional
        Labels for nodes. When supplied, :meth:`transform` returns a DataFrame
        whose columns preserve these labels. If omitted for DataFrame input,
        labels are inferred from its columns.
    state_labels : list-like or MultiIndex-like, optional
        Labels for states. When supplied, :meth:`transform` returns a DataFrame
        whose row index identifies each transition endpoint. If omitted for
        DataFrame input, labels are inferred from its index.
    store_trajectories : bool, default=True
        Whether to retain state trajectories from the latest transform call.
    store_control_trajectories : bool, default=True
        Whether to retain control trajectories from the latest transform call.
    """

    def __init__(
        self,
        A,
        T,
        B="identity",
        rho=1,
        S="identity",
        energy_type="optimal",
        order="permutations",
        system="continuous",
        xr="zero",
        expm_version="scipy",
        masker=None,
        memory=None,
        memory_level=1,
        verbose=0,
        normalize_A=True,
        c=1,
        node_labels=None,
        state_labels=None,
        store_trajectories=True,
        store_control_trajectories=True,
    ):
        self.A = A
        self.T = T
        self.B = B
        self.rho = rho
        self.S = S
        self.energy_type = energy_type
        self.order = order
        self.system = system
        self.xr = xr
        self.expm_version = expm_version
        self.masker = masker
        self.memory = memory
        self.memory_level = memory_level
        self.verbose = verbose
        self.normalize_A = normalize_A
        self.c = c
        self.node_labels = node_labels
        self.state_labels = state_labels
        self.store_trajectories = store_trajectories
        self.store_control_trajectories = store_control_trajectories
        
    def _fit_states(self, X):
        if _is_state_matrix(X):
            self.masker_ = None
            return _as_2d_float_array(X, "X")
        if self.masker is None:
            raise ValueError(
                "Image-like X requires a masker, for example "
                "NiftiLabelsMasker(labels_img=atlas)"
            )
        self.masker_ = clone(self.masker)
        return _as_2d_float_array(self.masker_.fit_transform(X), "masked X")

    def _transform_states(self, X):
        if _is_state_matrix(X):
            return _as_2d_float_array(X, "X")
        if self.masker_ is None:
            raise ValueError("This transformer was fitted without an image masker")
        return _as_2d_float_array(self.masker_.transform(X), "masked X")

    def fit(self, X=None, y=None, *, x0=None, xf=None):
        """Fit the optional image masker and validate the transition inputs."""
        self._fit_cache()
        state_input = _resolve_state_input(X=X, x0=x0, xf=xf)
        states = self._fit_states(state_input) # FIXME: Why not safe states as self.states_?
        A, T, B, _, rho, S, c = _validate_transition_inputs(
            self.A,
            self.T,
            self.B,
            states,
            self.rho,
            self.S,
            self.energy_type,
            self.order,
            self.system,
            normalize_A=self.normalize_A,
            c=self.c,
        )
        self.c_ = c
        self.A_ = A
        self.T_ = T
        self.B_ = B
        self.S_ = S
        self.rho_ = rho
        self.n_features_in_ = states.shape[1]
        self.n_states_in_ = states.shape[0]
        self.store_trajectories_ = _validate_boolean(
            self.store_trajectories, "store_trajectories"
        )
        self.store_control_trajectories_ = _validate_boolean(
            self.store_control_trajectories,
            "store_control_trajectories",
        )
        dataframe_node_labels = (
            state_input.columns if isinstance(state_input, pd.DataFrame) else None
        )
        dataframe_state_labels = (
            state_input.index if isinstance(state_input, pd.DataFrame) else None
        )
        node_labels = (
            self.node_labels
            if self.node_labels is not None
            else dataframe_node_labels
        )
        state_labels = (
            self.state_labels
            if self.state_labels is not None
            else dataframe_state_labels
        )
        self.node_labels_ = (
            None
            if node_labels is None
            else _coerce_labels(node_labels, states.shape[1], "node_labels")
        )
        self.state_labels_ = (
            None
            if state_labels is None
            else _coerce_labels(state_labels, states.shape[0], "state_labels")
        )
        return self

    def transform(self, X=None, *, x0=None, xf=None):
        """Return integrated control energy with shape transitions by nodes."""
        
        # FIXME: See above. Why not assign self.states_ in .fit()? Then we don't have
        # to check again if the state input is valid, and we would not have to again
        # call:
        #    state_input = _resolve_state_input(X=X, x0=x0, xf=xf)
        #    states = self._fit_states(state_input) 
        check_is_fitted(self, attributes=["A_", "B_", "S_", "masker_"])
        state_input = _resolve_state_input(X=X, x0=x0, xf=xf)
        states = self._transform_states(state_input)
        if states.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {states.shape[1]} nodes, but fit saw {self.n_features_in_}"
            )
        output_node_labels = self.node_labels_
        output_state_labels = self.state_labels_
        if isinstance(state_input, pd.DataFrame):
            if self.node_labels is None:
                output_node_labels = _coerce_labels(
                    state_input.columns, states.shape[1], "node_labels"
                )
            if self.state_labels is None:
                output_state_labels = _coerce_labels(
                    state_input.index, states.shape[0], "state_labels"
                )
        if output_state_labels is not None and len(
            output_state_labels
        ) != states.shape[0]:
            raise ValueError(
                "state_labels must contain one value per transformed state; "
                f"got {len(output_state_labels)} values for "
                f"{states.shape[0]} states"
            )

        transition_order = self.order
        cached_transition = self._cache(
            state_to_state_transition, func_memory_level=1
        )

        trajectories, control_trajectories, self.errors_ = cached_transition(
            A=self.A_,
            T=self.T_,
            B=self.B_,
            X=states,
            rho=self.rho_,
            S=self.S_,
            energy_type=self.energy_type,
            order=transition_order,
            system=self.system,
            xr=self.xr,
            expm_version=self.expm_version,
        )
        if self.store_trajectories_:
            self.trajectories_ = trajectories
        else:
            self.__dict__.pop("trajectories_", None)
        if self.store_control_trajectories_:
            self.control_trajectories_ = control_trajectories
        else:
            self.__dict__.pop("control_trajectories_", None)
        _, indices = _set_transition_order(states.shape[0], transition_order)
        self.transition_indices_ = [
            (source, target) for _, source, target in indices
        ]
        self.energies_ = state_to_state_integration(control_trajectories)
        if output_node_labels is None and output_state_labels is None:
            self.__dict__.pop("energies_df_", None)
            return self.energies_
        self.energies_df_ = get_state_to_state_df(
            self.energies_,
            order=transition_order,
            node_labels=output_node_labels,
            state_labels=output_state_labels,
        )
        return self.energies_df_

    # TODO: Do we really need to define this? Because this should automatically be
    # available by inheriting from TransformerMixin
    def fit_transform(self, X=None, y=None, *, x0=None, xf=None):
        """Fit and transform states supplied as ``X`` or as ``x0`` and ``xf``."""
        return self.fit(X, y, x0=x0, xf=xf).transform(
            X, x0=x0, xf=xf
        )

    def get_errors(self):
        """Return numerical errors from the most recent transform call."""
        check_is_fitted(self, attributes=["errors_"])
        return self.errors_.copy()

    # TODO: Should be split up into .get_state_trajectories and .get_control_trajectories
    # TODO: Should also output pandas dataframes with shape (time,nodes,transition)
    def get_transition_arrays(self):
        """Return retained trajectory arrays, or ``None`` when disabled."""
        check_is_fitted(self, attributes=["errors_"])
        trajectories = getattr(self, "trajectories_", None)
        control_trajectories = getattr(self, "control_trajectories_", None)
        return (
            None if trajectories is None else trajectories.copy(),
            (
                None
                if control_trajectories is None
                else control_trajectories.copy()
            ),
        )

    def get_feature_names_out(self, input_features=None):
        """Return names for the node-level energy columns."""
        check_is_fitted(self, attributes=["n_features_in_"])
        if input_features is not None:
            names = _coerce_labels(
                input_features, self.n_features_in_, "input_features"
            )
        elif self.node_labels_ is not None:
            names = self.node_labels_
        else:
            names = [
                f"node_{index}" for index in range(self.n_features_in_)
            ]
        result = np.empty(self.n_features_in_, dtype=object)
        result[:] = list(names)
        return result

__all__ = [
    "Transitioner",
    "get_state_to_state_df",
    "state_to_state_integration",
    "state_to_state_transition",
]
