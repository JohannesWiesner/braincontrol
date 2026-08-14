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


_TRANSITION_ORDERS = ("combinations", "permutations", "product", "stability")
_ENERGY_TYPES = ("minimal", "optimal")

########################################################################################
## Input validation
########################################################################################


def _as_2d_float_array(value, name):
    """Validate that input matrix is 2D and has only finited values"""
    array = np.asarray(value, dtype=float)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional array")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array

def _resolve_square_matrix(value, name, n_nodes):
    """Validate that input matrix is square-like and has number of nodes"""
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


def _validate_transition_inputs(A, T, B, X, rho, S, energy_type, order, system):
    """Validate all inputs to compute control energy"""

    A = _as_2d_float_array(A, "A")
    if A.shape[0] != A.shape[1]:
        raise ValueError(f"A must be square; got shape {A.shape}")

    X = _as_2d_float_array(X, "X")
    n_nodes = A.shape[0]
    if X.shape[1] != n_nodes:
        raise ValueError(
            "X must have one column per node in A; "
            f"got {X.shape[1]} columns for {n_nodes} nodes"
        )

    if order not in _TRANSITION_ORDERS:
        choices = ", ".join(repr(choice) for choice in _TRANSITION_ORDERS)
        raise ValueError(f"order must be one of {choices}; got {order!r}")
    if energy_type not in _ENERGY_TYPES:
        choices = ", ".join(repr(choice) for choice in _ENERGY_TYPES)
        raise ValueError(
            f"energy_type must be one of {choices}; got {energy_type!r}"
        )
    if system not in ("continuous", "discrete"):
        raise ValueError("system must be 'continuous' or 'discrete'")
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
    return A, T, B, X, rho, S

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
    if order not in _TRANSITION_ORDERS:
        choices = ", ".join(repr(choice) for choice in _TRANSITION_ORDERS)
        raise ValueError(f"order must be one of {choices}; got {order!r}")

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


def state_to_state_transition(
    A,
    T,
    B,
    X,
    rho=1,
    S="identity",
    energy_type="optimal",
    order="permutations",
    *,
    system="continuous",
    xr="zero",
    expm_version="scipy",
):
    """Compute trajectories and control inputs between every requested state.

    Parameters
    ----------
    A : array-like of shape (n_nodes, n_nodes)
        A normalized adjacency matrix.
    T : float
        Time horizon.
    B : array-like of shape (n_nodes, n_nodes) or ``"identity"``
        Control input matrix.
    X : array-like of shape (n_states, n_nodes)
        One state per row.
    rho : float or None, default=1
        Mixing parameter used by :func:`nctpy.energies.get_control_inputs`.
        Must be ``None`` for minimal energy and positive for optimal energy.
    S : array-like of shape (n_nodes, n_nodes), ``"identity"``, or None
        State-trajectory constraint matrix. Must be ``None`` for minimal
        energy and provided for optimal energy.
    energy_type : {"minimal", "optimal"}, default="optimal"
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
    control_inputs : ndarray of shape (n_control_points, n_nodes, n_transitions)
    errors : ndarray of shape (n_transitions, 2)
        Numerical errors reported by ``nctpy`` for each transition.
    """
    A, T, B, X, rho, S = _validate_transition_inputs(
        A, T, B, X, rho, S, energy_type, order, system
    )
    n_transitions, transition_indices = _set_transition_order(X.shape[0], order)

    if energy_type == "minimal":
        solver_rho = 1
        solver_S = np.zeros_like(A)
    else:
        solver_rho = rho
        solver_S = S

    trajectories = []
    control_inputs = []
    errors = []
    for _, source, target in transition_indices:
        trajectory, control_input, error = get_control_inputs(
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
        control_inputs.append(control_input)
        errors.append(error)

    if n_transitions:
        trajectory_array = np.stack(trajectories, axis=2)
        control_input_array = np.stack(control_inputs, axis=2)
        error_array = np.asarray(errors, dtype=float)
    else:
        if system == "continuous":
            n_trajectory_points = round(T / 0.001) + 1
            n_control_points = n_trajectory_points
        else:
            n_trajectory_points = int(T) + 1
            n_control_points = int(T)
        trajectory_array = np.empty((n_trajectory_points, A.shape[0], 0))
        control_input_array = np.empty((n_control_points, A.shape[0], 0))
        error_array = np.empty((0, 2))

    return trajectory_array, control_input_array, error_array

# FIXME: Should be renamed to state_to_state_integration
def state_to_state_aggregation(control_inputs):
    """Integrate squared control inputs over time for every transition."""
    control_inputs = np.asarray(control_inputs, dtype=float)
    if control_inputs.ndim != 3:
        raise ValueError(
            "control_inputs must have shape "
            "(n_time_points, n_nodes, n_transitions)"
        )
    if not np.all(np.isfinite(control_inputs)):
        raise ValueError("control_inputs must contain only finite values")

    n_nodes, n_transitions = control_inputs.shape[1:]
    energies = np.empty((n_transitions, n_nodes))
    for transition in range(n_transitions):
        energies[transition] = integrate_u(control_inputs[:, :, transition])
    return energies


def get_transition_info(state_attributes, order):
    """Pair state attributes in the same order as the requested transitions."""
    attributes = list(state_attributes)
    _, transition_indices = _set_transition_order(len(attributes), order)
    return [
        (attributes[source], attributes[target])
        for _, source, target in transition_indices
    ]


def _coerce_attributes(attributes, expected_length, parameter_name):
    """Return list-like attributes as a pandas Index or MultiIndex."""
    if isinstance(attributes, (str, bytes, Mapping, pd.DataFrame)):
        raise TypeError(
            f"{parameter_name} must be a list-like or MultiIndex-like object"
        )

    if isinstance(attributes, pd.MultiIndex):
        index = attributes.copy()
    else:
        try:
            values = list(attributes)
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
            names = getattr(attributes, "names", None)
            index = pd.MultiIndex.from_tuples(values, names=names)
        else:
            name = getattr(attributes, "name", None)
            index = pd.Index(values, name=name)

    if expected_length is not None and len(index) != expected_length:
        raise ValueError(
            f"{parameter_name} must contain {expected_length} values; "
            f"got {len(index)}"
        )
    if len(index) == 0:
        raise ValueError(f"{parameter_name} must contain at least one value")
    return index


def _attribute_level_names(
    attributes, default_name, attribute_prefix=None, reserved=()
):
    """Return unique, non-conflicting names for attribute levels."""
    if isinstance(attributes, pd.MultiIndex):
        source_names = attributes.names
    else:
        source_names = [attributes.name]

    names = []
    reserved = set(reserved)
    attribute_prefix = attribute_prefix or default_name
    for level, source_name in enumerate(source_names):
        fallback = (
            default_name
            if len(source_names) == 1
            else f"{attribute_prefix}_attribute_{level}"
        )
        name = source_name
        if not isinstance(name, str) or not name or name in reserved or name in names:
            name = fallback
        names.append(name)
    return names

# TODO: This should not flatten multiindeces, but instead add another higher order level (transition)
# that maps the lower order levels to "source" and "target"
def _state_transition_columns(df, state_attributes, order, n_transitions):
    """Add source and target columns for every state attribute level."""
    state_attributes = _coerce_attributes(
        state_attributes, None, "state_attributes"
    )
    expected_transitions, transition_indices = _set_transition_order(
        len(state_attributes), order
    )
    if expected_transitions != n_transitions:
        raise ValueError(
            "state_to_state_array rows do not match state_attributes and order"
        )

    level_names = _attribute_level_names(
        state_attributes, "state", reserved={"transition_name", "value"}
    )
    if isinstance(state_attributes, pd.MultiIndex):
        level_values = [
            state_attributes.get_level_values(level).to_numpy()
            for level in range(state_attributes.nlevels)
        ]
    else:
        level_values = [state_attributes.to_numpy()]

    transition_columns = []
    for level_name, values in zip(level_names, level_values):
        source_column = f"source_{level_name}"
        target_column = f"target_{level_name}"
        df[source_column] = [values[source] for _, source, _ in transition_indices]
        df[target_column] = [values[target] for _, _, target in transition_indices]
        transition_columns.extend([source_column, target_column])

    source_name, target_name = transition_columns[-2:]
    df["transition_name"] = [
        f"{source}-{target}"
        for source, target in zip(df[source_name], df[target_name])
    ]
    transition_columns.append("transition_name")
    return transition_columns


def get_state_to_state_df(
    state_to_state_array,
    order,
    node_attributes=None,
    state_attributes=None,
):
    """Create a labelled DataFrame from a transition-by-node array.

    Parameters
    ----------
    state_to_state_array : array-like of shape (n_transitions, n_nodes)
        Values produced for every transition and node.
    order : {"combinations", "permutations", "product", "stability"}
        Ordering used to construct the transitions.
    node_attributes : list-like or MultiIndex-like, optional
        One label or tuple of hierarchical labels per node. A pandas
        ``MultiIndex`` is preserved directly. A list-like object containing
        equal-length tuples is converted to a ``MultiIndex``. Other list-like
        values become a regular ``Index``. This is the only parameter for node
        metadata.
    state_attributes : list-like or MultiIndex-like, optional
        One label or tuple of hierarchical labels per state. A pandas
        ``MultiIndex`` is expanded into paired ``source_<level>`` and
        ``target_<level>`` transition fields. Tuple-based list-like values are
        converted to a ``MultiIndex``. Flat values produce the traditional
        ``source_state`` and ``target_state`` fields. This is the only
        parameter for state metadata.
    Returns
    -------
    pandas.DataFrame
        A transition-by-node table.
    """
    values = _as_2d_float_array(state_to_state_array, "state_to_state_array")
    n_transitions, n_nodes = values.shape
    if node_attributes is None:
        node_attributes = pd.RangeIndex(n_nodes)
    else:
        node_attributes = _coerce_attributes(
            node_attributes, n_nodes, "node_attributes"
        )

    df = pd.DataFrame(values)
    transition_columns = []

    if state_attributes is not None:
        transition_columns = _state_transition_columns(
            df, state_attributes, order, n_transitions
        )

    if transition_columns:
        df = df.set_index(transition_columns)
    df.columns = node_attributes
    return df

def get_transition_df(
    A,
    T,
    B,
    X,
    rho=1,
    S="identity",
    energy_type="optimal",
    order="permutations",
    **kwargs,
):
    """Compute integrated transition energy and return it as a DataFrame."""
    transition_keywords = {
        key: kwargs.pop(key)
        for key in ("system", "xr", "expm_version")
        if key in kwargs
    }
    _, control_inputs, _ = state_to_state_transition(
        A=A,
        T=T,
        B=B,
        X=X,
        rho=rho,
        S=S,
        energy_type=energy_type,
        order=order,
        **transition_keywords,
    )
    energies = state_to_state_aggregation(control_inputs)
    return get_state_to_state_df(energies, order=order, **kwargs)

# TODO: Not sure if there is a better way to do this. The general goal here is
# that Transitioner should be able to compute also other things than control energy
# for a given source-target state pair.
def state_to_state_comparison(X, func, order="permutations"):
    """Apply a pairwise operation to requested source and target states."""
    X = _as_2d_float_array(X, "X")
    n_transitions, transition_indices = _set_transition_order(X.shape[0], order)

    if func == "difference":
        operation = np.subtract
    elif func == "sum":
        operation = np.add
    elif callable(func):
        operation = func
    else:
        raise ValueError("func must be 'difference', 'sum', or a callable")

    comparisons = np.empty((n_transitions, X.shape[1]))
    for transition, source, target in transition_indices:
        result = np.asarray(operation(X[source], X[target]), dtype=float)
        if result.shape != (X.shape[1],):
            raise ValueError(
                "func must return one value per node; "
                f"got shape {result.shape}"
            )
        comparisons[transition] = result
    return comparisons

# TODO: Not sure if this is needed, because get_state_to_state_df can also handle this?
def get_state_comparison_df(X, func, order="permutations", **kwargs):
    """Apply a pairwise state operation and return a labelled DataFrame."""
    comparisons = state_to_state_comparison(X, func=func, order=order)
    return get_state_to_state_df(comparisons, order=order, **kwargs)

# FIXME: Update docstring
# FIXME: Add node_attributes and state_attributes
class Transitioner(
    TransformerMixin, CacheMixin, BaseEstimator, auto_wrap_output_keys=None
):
    """Transform states or labelled NIfTI images into transition energies.

    ``X`` passed to :meth:`fit` and :meth:`transform` can be a two-dimensional
    ``(n_states, n_nodes)`` array.  For image-like input, pass a compatible
    masker such as ``NiftiLabelsMasker(labels_img=atlas)`` to the constructor;
    each volume in a 4D image is then treated as one state.

    The fitted transformer exposes unintegrated results through
    :meth:`get_transition_arrays` and numerical errors through
    :meth:`get_errors`.

    State input can be provided either as ``X``, containing one state per row,
    or as the separate ``x0`` and ``xf`` keyword arguments. The latter always
    computes the single directed transition from ``x0`` to ``xf``.

    Parameters
    ----------
    energy_type : {"minimal", "optimal"}, default="optimal"
        Type of control energy to compute. ``"minimal"`` is mutually exclusive
        with ``rho`` and ``S``, so both must be ``None``. ``"optimal"`` requires
        both parameters to be provided.
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

    @staticmethod
    def _is_state_matrix(X):
        try:
            array = np.asarray(X)
        except Exception:
            return False
        return array.ndim == 2 and np.issubdtype(array.dtype, np.number)

    # FIXME: Input validation should happen in _validate_transition_inputs if possible
    @staticmethod
    def _resolve_state_input(X=None, x0=None, xf=None):
        """Return state input and whether it represents one explicit pair."""
        endpoints_provided = x0 is not None or xf is not None
        if X is not None and endpoints_provided:
            raise ValueError("Provide either X or x0 and xf, not both")
        if X is not None:
            return X, False
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
            is_numeric = array is not None and np.issubdtype(
                array.dtype, np.number
            )
            endpoint_is_numeric.append(is_numeric)
            if is_numeric:
                if array.ndim != 1:
                    raise ValueError(f"{name} must be a one-dimensional state")
                endpoint_arrays.append(array)

        if all(endpoint_is_numeric):
            if endpoint_arrays[0].shape != endpoint_arrays[1].shape:
                raise ValueError("x0 and xf must contain the same number of nodes")
            return np.stack(endpoint_arrays), True
        if any(endpoint_is_numeric):
            raise TypeError("x0 and xf must use the same input type")
        return [x0, xf], True

    def _fit_states(self, X):
        if self._is_state_matrix(X):
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
        if self._is_state_matrix(X):
            return _as_2d_float_array(X, "X")
        if self.masker_ is None:
            raise ValueError("This transformer was fitted without an image masker")
        return _as_2d_float_array(self.masker_.transform(X), "masked X")

    def fit(self, X=None, y=None, *, x0=None, xf=None):
        """Fit the optional image masker and validate the transition inputs."""
        self._fit_cache()
        state_input, _ = self._resolve_state_input(X=X, x0=x0, xf=xf)
        states = self._fit_states(state_input)

        # FIXME: Input validation should happen in _validate_transition_inputs if possible
        if not isinstance(self.normalize_A, (bool, np.bool_)):
            raise TypeError("normalize_A must be a boolean")
        if (
            isinstance(self.c, (bool, np.bool_)) # FIXME:c must not be a bool
            or not isinstance(self.c, (int, float, np.integer, np.floating))
            or not np.isfinite(self.c)
            or self.c <= 0
        ):
            raise ValueError("c must be a positive finite number")
        self.c_ = float(self.c)
        adjacency = _as_2d_float_array(self.A, "A")
        # FIXME: Input validation should happen in _validate_transition_inputs if possible
        if adjacency.shape[0] != adjacency.shape[1]:
            raise ValueError(f"A must be square; got shape {adjacency.shape}")
        if self.system not in ("continuous", "discrete"):
            raise ValueError("system must be 'continuous' or 'discrete'")
        if self.normalize_A:
            adjacency = matrix_normalization(
                adjacency, system=self.system, c=self.c_
            )
        A, T, B, _, rho, S = _validate_transition_inputs(
            adjacency,
            self.T,
            self.B,
            states,
            self.rho,
            self.S,
            self.energy_type,
            self.order,
            self.system,
        )
        self.A_ = A
        self.T_ = T
        self.B_ = B
        self.S_ = S
        self.rho_ = rho
        self.n_features_in_ = states.shape[1]
        self.n_states_in_ = states.shape[0]
        return self

    def transform(self, X=None, *, x0=None, xf=None):
        """Return integrated control energy with shape transitions by nodes."""
        check_is_fitted(self, attributes=["A_", "B_", "S_", "masker_"])
        state_input, explicit_pair = self._resolve_state_input(
            X=X, x0=x0, xf=xf
        )
        states = self._transform_states(state_input)
        if states.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {states.shape[1]} nodes, but fit saw {self.n_features_in_}"
            )

        transition_order = "combinations" if explicit_pair else self.order
        cached_transition = self._cache(
            state_to_state_transition, func_memory_level=1
        )
        self.trajectories_, self.control_inputs_, self.errors_ = (
            cached_transition(
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
        )
        _, indices = _set_transition_order(states.shape[0], transition_order)
        self.transition_indices_ = [
            (source, target) for _, source, target in indices
        ]
        self.energies_ = state_to_state_aggregation(self.control_inputs_)
        return self.energies_

    def fit_transform(self, X=None, y=None, *, x0=None, xf=None):
        """Fit and transform states supplied as ``X`` or as ``x0`` and ``xf``."""
        return self.fit(X, y, x0=x0, xf=xf).transform(
            X, x0=x0, xf=xf
        )

    def get_errors(self):
        """Return numerical errors from the most recent transform call."""
        check_is_fitted(self, attributes=["errors_"])
        return self.errors_.copy()

    def get_transition_arrays(self):
        """Return trajectory and control-input arrays from the latest transform."""
        check_is_fitted(self, attributes=["trajectories_", "control_inputs_"])
        return self.trajectories_.copy(), self.control_inputs_.copy()

    def get_feature_names_out(self, input_features=None):
        """Return names for the node-level energy columns."""
        check_is_fitted(self, attributes=["n_features_in_"])
        if input_features is not None:
            input_features = np.asarray(input_features, dtype=object)
            if len(input_features) != self.n_features_in_:
                raise ValueError(
                    "input_features must contain one name per fitted node"
                )
            return input_features
        return np.asarray(
            [f"node_{index}" for index in range(self.n_features_in_)],
            dtype=object,
        )

__all__ = [
    "Transitioner",
    "get_state_comparison_df",
    "get_state_to_state_df",
    "get_transition_df",
    "get_transition_info",
    "state_to_state_aggregation",
    "state_to_state_comparison",
    "state_to_state_transition",
]
