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
from nilearn.image import check_niimg
from nilearn.image import concat_imgs

###############################################################################
## Validation of Network Control Theory parameters
###############################################################################

def _validate_2d_matrix_and_finite(value, name):
    """Validate that input is two-dimensional matrix and contains only finite values."""
    array = np.asarray(value)

    if array.ndim != 2:
        raise ValueError(f"{name} must be two-dimensional")

    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")

def _validate_square_matrix_or_identity(value, name):
    """Validate that input is a finite square array-like object or 'identity'."""
    if isinstance(value, str):
        if value != "identity":
            raise ValueError(
                f"{name} must be a square array-like object or 'identity'"
            )
        return

    if not isinstance(value, (np.ndarray, pd.DataFrame)):
        raise TypeError(
            f"{name} must be a NumPy array, pandas DataFrame, or 'identity'"
        )

    _validate_2d_matrix_and_finite(value, name)

    if value.shape[0] != value.shape[1]:
        raise ValueError(
            f"{name} must be square; got shape {value.shape}"
        )

def _resolve_array_or_identity(value, n_nodes):
    """Return the input array-like object as numpy array or as an identity matrix of size n_nodes."""
    if isinstance(value, str):
        return np.eye(n_nodes)

    return np.asarray(value)

def _validate_same_shape(arrays, names):
    """Validate that all arrays have the same shape."""
    shapes = {array.shape for array in arrays}

    if len(shapes) != 1:
        formatted_shapes = ", ".join(
            f"{name}: {array.shape}"
            for name, array in zip(names, arrays)
        )
        raise ValueError(
            f"{', '.join(names)} must have the same shape; "
            f"got {formatted_shapes}"
        )

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

def _validate_adjacency_inputs(A, normalize_A, c):
    """Validate adjacency matrix and normalization parameters.
    ``A`` must be two-dimensional, square, and contain only finite values.
    ``normalize_A`` must be boolean.
    ``c`` must be a positive finite real number.
    """
    if not isinstance(A, (np.ndarray, pd.DataFrame)):
        raise TypeError(
            "A must be a NumPy array or pandas DataFrame"
        )
        
    _validate_2d_matrix_and_finite(A, "A")

    if A.shape[0] != A.shape[1]:
        raise ValueError(
            f"A must be square; got shape {A.shape}"
        )

    _validate_boolean(normalize_A, "normalize_A")

    if (
        isinstance(c, (bool, np.bool_))
        or not isinstance(c, (int, float, np.integer, np.floating))
    ):
        raise TypeError("c must be a real number")

    if not np.isfinite(c) or c <= 0:
        raise ValueError("c must be a positive finite number")

# NOTE: It would be better if nctpy itself would offer such validation functions
# then we could import them.
def _validate_transition_inputs(
    A,
    T,
    B,
    rho,
    S,
    energy_type,
    system,
    xr,
    expm_version,
    normalize_A,
    c
):
    """Validate inputs used to compute control energy."""

    # check categorical parameters
    energy_type = _validate_choice(
        energy_type,
        "energy_type",
        ("minimal", "optimal"),
    )

    system = _validate_choice(
        system,
        "system",
        ("continuous", "discrete"),
    )
    xr = _validate_choice(
        xr,
        "xr",
        ("x0", "xf", "zero", "midpoint"),
    )
    expm_version = _validate_choice(
        expm_version,
        "expm_version",
        ("scipy", "eig"),
    )

    # check adjacency matrix and normalization parameters
    _validate_adjacency_inputs(A,normalize_A,c)

    # check B matrix
    _validate_square_matrix_or_identity(B, "B")

    # check energy-specific parameters
    # FIXME: check what rho has to be when energy_type is minimal
    if energy_type == "minimal":
        if rho is not None or S is not None:
            raise ValueError(
                "rho and S must both be None when energy_type='minimal'"
            )

    elif energy_type == 'optimal':
        if rho is None or S is None:
            raise ValueError(
                "rho and S must both be provided when energy_type='optimal'"
            )
        
        # check rho if energy_type is optimal
        if (
            isinstance(rho,(bool, np.bool_))
            or not isinstance(rho, (float, np.floating))
            or not np.isfinite(rho)
            or rho <= 0.0
            or rho > 1.0
        ):
            raise ValueError("rho must be a positive finite float between 0 and 1")

        # check S matrix if energy_type is optimal
        _validate_square_matrix_or_identity(S,"S")

    # check time horizon
    if isinstance(T, (bool, np.bool_)) or not np.isscalar(T):
        raise TypeError("T must be a scalar number")

    if not np.isfinite(T) or T <= 0:
        raise ValueError("T must be a positive finite number")

    if system == "discrete":
        if not isinstance(T, (int, np.integer)) or T < 2:
            raise ValueError(
                "T must be an integer of at least 2 for a discrete system"
            )
    elif not isinstance(T, (float, np.floating)):
        raise TypeError(
            "T must be a float for a continuous system"
        )

###############################################################################
## Validation of states
###############################################################################

def _is_niimg_or_tabular_like(value):
    """Determine whether input is niimg-like or tabular-like. The order of
    checks is important here because some niimg-like objects can still behave like
    tabular-like objects.

    Returns
    -------
    {"tabular_like", "niimg_like"}
        The input type.

    Raises
    ------
    TypeError
        If the input is neither tabular-like nor Niimg-like.
    """
    try:
        check_niimg(value)
        return "niimg_like"
    except (TypeError, ValueError):
        pass
    
    try:
        np.asarray(value)
        return "tabular_like"
    except (TypeError, ValueError):
        pass

    raise TypeError("Input must be a Niimg-like or tabular-like object")

def _validate_single_state_niimg(value, name):
    """Validate that Niimg-like input represents exactly one state."""
    img = check_niimg(value)

    if img.ndim == 3:
        return img

    if img.ndim == 4 and img.shape[3] == 1:
        return img

    raise ValueError(
        f"{name} must represent exactly one state; "
        f"got image shape {img.shape}"
    )

# TODO: Maybe this could also output the output type then we would not
# have to call _is_niimg_or_tabular_like again
def _resolve_state_input(X=None, x0=None, xf=None):
    """Validate state input and return it in a consistent representation.

    States are represented by rows and nodes by columns for tabular input,
    and by volumes along the fourth dimension for Niimg-like input.

    Exactly one of the following must be provided:

    - ``X``: a two-dimensional NumPy array, pandas DataFrame, or Niimg-like
      object containing one or more states.
    - ``x0`` and ``xf``: two one-dimensional NumPy arrays, two pandas Series,
      or two Niimg-like objects representing one state each.

    Tabular input is returned as a DataFrame with shape
    ``(n_states, n_nodes)``. Niimg-like input is returned as a 4D Niimg-like
    object with one volume per state.

    Parameters
    ----------
    X : np.ndarray, pd.DataFrame, or Niimg-like, optional
        State input. For tabular input, rows represent states and columns
        represent nodes. Niimg-like input may contain one or more states.
    x0 : np.ndarray, pd.Series, or Niimg-like, optional
        Initial state. Niimg-like input must represent exactly one state.
    xf : np.ndarray, pd.Series, or Niimg-like, optional
        Final state. Niimg-like input must represent exactly one state.

    Returns
    -------
    pd.DataFrame or Niimg-like
        Resolved state input. Tabular input is returned as a DataFrame.
        Niimg-like input is returned as a 4D image.
    """
    endpoints_provided = x0 is not None or xf is not None

    if X is not None and endpoints_provided:
        raise ValueError("Provide either X or x0 and xf, not both")

    if X is None and x0 is None and xf is None:
        raise ValueError("Provide either X or both x0 and xf")

    # X contains all states
    if X is not None:
        X_type = _is_niimg_or_tabular_like(X)

        if X_type == "niimg_like":
            # concat_imgs guarantees a 4D representation.
            return concat_imgs([X])

        if not isinstance(X, (np.ndarray, pd.DataFrame)):
            raise TypeError(
                "X must be a NumPy array, pandas DataFrame, "
                "or Niimg-like object"
            )

        _validate_2d_matrix_and_finite(X, "X")

        if isinstance(X, pd.DataFrame):
            return X.copy()

        return pd.DataFrame(X)

    # x0 and xf must be provided together
    if x0 is None or xf is None:
        raise ValueError("x0 and xf must be provided together")

    x0_type = _is_niimg_or_tabular_like(x0)
    xf_type = _is_niimg_or_tabular_like(xf)

    if x0_type != xf_type:
        raise TypeError("x0 and xf must use the same input type")

    # Niimg-like endpoints
    if x0_type == "niimg_like":
        x0_img = _validate_single_state_niimg(x0, "x0")
        xf_img = _validate_single_state_niimg(xf, "xf")

        return concat_imgs([x0_img, xf_img])

    # Tabular endpoints must use the same concrete representation.
    if type(x0) is not type(xf):
        raise TypeError("x0 and xf must use the same input type")

    if isinstance(x0, np.ndarray):
        if x0.ndim != 1 or xf.ndim != 1:
            raise ValueError(
                "x0 and xf must be one-dimensional states"
            )

        if x0.shape != xf.shape:
            raise ValueError(
                "x0 and xf must contain the same number of nodes"
            )

        if x0.dtype != xf.dtype:
            raise TypeError(
                "x0 and xf must have the same dtype"
            )

        if not np.all(np.isfinite(x0)) or not np.all(np.isfinite(xf)):
            raise ValueError(
                "x0 and xf must contain only finite values"
            )

        return pd.DataFrame(
            np.stack((x0, xf))
        )

    if isinstance(x0, pd.Series):
        if not x0.index.equals(xf.index):
            raise ValueError(
                "x0 and xf must have matching indices"
            )

        if x0.dtype != xf.dtype:
            raise TypeError(
                "x0 and xf must have the same dtype"
            )

        if not np.all(np.isfinite(x0)) or not np.all(np.isfinite(xf)):
            raise ValueError(
                "x0 and xf must contain only finite values"
            )

        return pd.DataFrame(
            [x0.to_numpy(), xf.to_numpy()],
            columns=x0.index,
        )

    raise TypeError(
        "x0 and xf must both be NumPy arrays, both be pandas Series, "
        "or both be Niimg-like objects"
    )
    
    
def _validate_transition_order(n_states, order):
    """Validate that the requested transition order is possible."""
    order = _validate_choice(
        order,
        "order",
        ("combinations", "permutations", "product", "stability"),
    )

    if n_states < 1:
        raise ValueError("State input must contain at least one state")

    if n_states == 1 and order in ("combinations", "permutations"):
        raise ValueError(
            f"order={order!r} requires at least two states"
        )

    return order
    
###############################################################################
## Transitions
###############################################################################

# TODO: Find more suitable names for order choices
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
    
    n_transitions = len(transition_indices)
    
    return n_transitions, transition_indices

def get_transition_trajectories(
    A,
    X,
    T,
    B,
    rho,
    S,
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
    X : ndarray of shape (n_states, n_nodes)
        Validated state matrix with one state per row.
    T : float
        Time horizon.
    B : ndarray of shape (n_nodes, n_nodes)
        Validated control input matrix.
    rho : float or None
        Mixing parameter used by :func:`nctpy.energies.get_control_inputs`.
    S : ndarray of shape (n_nodes, n_nodes) or None
        Validated state-trajectory constraint matrix.
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
    state_trajectories : ndarray of shape (n_time_points, n_nodes, n_transitions)
    control_trajectories : ndarray of shape (n_time_points, n_nodes, n_transitions)
    errors : ndarray of shape (n_transitions, 2)
        Numerical errors reported by ``nctpy`` for each transition.
    """
    
    _, transition_indices = _set_transition_order(X.shape[0], order)

    # TODO: I think this can be made more memory efficient. The function
    # knows T so we can already define empty arrays that have the appropriate size
    state_trajectories = []
    control_trajectories = []
    errors = []
    for _, source, target in transition_indices:
        state_trajectory, control_trajectory, error = get_control_inputs(
            A_norm=A,
            T=T,
            B=B,
            x0=X[source],
            xf=X[target],
            system=system,
            rho=rho,
            S=S,
            xr=xr,
            expm_version=expm_version,
        )
        state_trajectories.append(state_trajectory)
        control_trajectories.append(control_trajectory)
        errors.append(error)
        
    trajectory_array = np.stack(state_trajectories, axis=2)
    control_trajectory_array = np.stack(control_trajectories, axis=2)
    error_array = np.asarray(errors, dtype=float)

    return trajectory_array, control_trajectory_array, error_array

def get_transition_energy(control_trajectories):
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

    n_nodes, n_transitions = control_trajectories.shape[1:]
    energies = np.empty((n_transitions, n_nodes))
    for transition in range(n_transitions):
        energies[transition] = integrate_u(
            control_trajectories[:, :, transition]
        )
    return energies

###############################################################################
## Node and State labels
###############################################################################

def _coerce_labels(labels, expected_length, parameter_name):
    """Convert and validate labels as a pandas Index or MultiIndex.

    List-like input is converted to a named Index. Existing Index names are
    preserved. MultiIndex input must have a name for every level. The number
    of labels must match ``expected_length`` when provided.
    """
    
    if isinstance(labels, (str, bytes, Mapping, pd.DataFrame)):
        raise TypeError(
            f"{parameter_name} must be a list-like, Index, or MultiIndex object"
        )

    if isinstance(labels, pd.MultiIndex):
        index = labels.copy()

        if any(name is None for name in index.names):
            raise ValueError(
                f"All levels of {parameter_name} must have a name"
            )

    elif isinstance(labels, pd.Index):
        index = labels.copy()

    else:
        try:
            values = list(labels)
        except TypeError as error:
            raise TypeError(
                f"{parameter_name} must be a list-like, Index, "
                "or MultiIndex object"
            ) from error

        default_name = {
            "state_labels": "state",
            "node_labels": "node",
        }.get(parameter_name, parameter_name)

        index = pd.Index(values, name=default_name)

    if len(index) == 0:
        raise ValueError(
            f"{parameter_name} must contain at least one value"
        )

    if expected_length is not None and len(index) != expected_length:
        raise ValueError(
            f"{parameter_name} must contain {expected_length} values; "
            f"got {len(index)}"
        )

    return index

def _state_transition_index(state_labels, order):
    """Build an index describing state transitions.

    Each label is replaced by a ``(source, target)`` pair according to
    ``order``. For MultiIndex input, the original levels and their names are
    preserved.
    """
    n_states = len(state_labels)
    _, transition_indices = _set_transition_order(
        n_states,
        order,
    )

    if isinstance(state_labels, pd.MultiIndex):
        level_names = state_labels.names
        level_values = [
            state_labels.get_level_values(level).to_numpy()
            for level in range(state_labels.nlevels)
        ]

        transition_values = [
            [
                (values[source], values[target])
                for _, source, target in transition_indices
            ]
            for values in level_values
        ]

        return pd.MultiIndex.from_arrays(
            transition_values,
            names=level_names,
        )

    transition_values = [
        (state_labels[source], state_labels[target])
        for _, source, target in transition_indices
    ]

    # Keep each (source, target) tuple as one scalar Index value.
    values = np.empty(len(transition_values), dtype=object)
    values[:] = transition_values

    return pd.Index(
        values,
        name=state_labels.name,
    )

###############################################################################
## Class
###############################################################################

# TODO: Is it true that xr can be also array like?
class Transitioner(TransformerMixin, CacheMixin, BaseEstimator, auto_wrap_output_keys=None):
    """Transform states or labelled NIfTI images into transition energies.

    ``X`` passed to :meth:`fit` and :meth:`transform` can be a two-dimensional
    ``(n_states, n_nodes)`` array.  For image-like input, pass a compatible
    masker such as ``NiftiLabelsMasker(labels_img=atlas)`` to the constructor;
    each volume in a 4D image is then treated as one state.

    The fitted transformer exposes retained unintegrated results through
    :meth:`get_transition_arrays` and numerical errors through
    :meth:`get_errors`.

    State input can be provided either as ``X``, containing one state per row,
    or as the separate ``x0`` and ``xf`` keyword arguments.
    
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
    store_state_trajectories : bool, default=False
        Whether to retain state trajectories from the latest transform call.
    store_control_trajectories : bool, default=False
        Whether to retain control trajectories from the latest transform call.
    """

    def __init__(
        self,
        A,
        T,
        normalize_A=True,
        c=1,
        B="identity",
        rho=1.0,
        S="identity",
        energy_type="optimal",
        system="continuous",
        xr="zero",
        expm_version="scipy",
        masker=None,
        memory=None,
        memory_level=1,
        verbose=0,

        store_state_trajectories=False,
        store_control_trajectories=False,
    ):
        self.A = A
        self.T = T
        self.normalize_A = normalize_A
        self.c = c
        self.B = B
        self.rho = rho
        self.S = S
        self.energy_type = energy_type
        self.system = system
        self.xr = xr
        self.expm_version = expm_version
        self.masker = masker
        self.memory = memory
        self.memory_level = memory_level
        self.verbose = verbose
        self.store_state_trajectories = store_state_trajectories
        self.store_control_trajectories = store_control_trajectories
    
    def _fit_matrices(self):
        '''fit matrices `A`, `B`, `S`. 
        As A is the only object provided in init, everything else will be valdiated against it'''
        
        # fit A
        self.A_ = np.asarray(self.A)
        self.n_nodes_ = self.A_.shape[0]
    
        # fit normalized A
        self.A_norm_ = (
            self.A_.copy()
            if not self.normalize_A
            else matrix_normalization(
                self.A_,
                self.system,
                self.c,
            )
        )

        # fit B which is either provided or 'identity'
        self.B_ = _resolve_array_or_identity(self.B,self.n_nodes_)
        
        # fit S which is either provided, or None, or 'identity'
        if self.energy_type == "minimal":
            self.S_ = np.zeros_like(self.A_)
        else:
            self.S_ = _resolve_array_or_identity(self.S,self.n_nodes_,)
        
        # all matrices must have same shape
        _validate_same_shape([self.A_,self.B_,self.S_],["A","B","S"])
        
    def _fit_masker(self, X):
        """Clone, fit, and validate the masker for image-like state input."""
        if self.masker is None:
            raise ValueError(
                "Image-like state input requires a masker, for example "
                "NiftiLabelsMasker(labels_img=atlas)"
            )
    
        self.masker_ = clone(self.masker)
        self.masker_.fit(X)
    
        required_attributes = ("n_elements_", "lut_")
        missing_attributes = [
            attribute
            for attribute in required_attributes
            if not hasattr(self.masker_, attribute)
        ]
    
        if missing_attributes:
            raise TypeError(
                "masker must expose the fitted attributes "
                f"{', '.join(required_attributes)}"
            )

    def _fit_states(self,X,x0,xf,node_labels):
        """Resolve state input and fit state-related metadata.
    
        Determine the number of states and nodes, fit the masker when required,
        and validate node and state labels against the resolved state input.
        
        node_labels : list-like or MultiIndex-like, optional
            Labels for nodes. When supplied, :meth:`transform` returns a DataFrame
            whose columns preserve these labels. If omitted labels are inferred from input.
        
        """
        
        # resolve state input, which will return either pandas dataframe or 4D-image
        self.X_ = _resolve_state_input(X=X,x0=x0,xf=xf)
        self.X_type_ = _is_niimg_or_tabular_like(self.X_)

        if self.X_type_ == "tabular_like":
            
            self.masker_ = None
            self.n_states_in_ = self.X_.shape[0]
            self.n_nodes_in_ = self.X_.shape[1]
            node_labels_inferred = self.X_.columns
    
        elif self.X_type_ == "niimg_like":
            
            self._fit_masker(self.X_)
            self.n_nodes_in_ = self.masker_.n_elements_
            self.n_states_in_ = self.X_.shape[3]

            lut = self.masker_.lut_
            lut = lut.loc[lut["index"] != self.masker_.background_label].reset_index(drop=True)
            node_labels_inferred = pd.MultiIndex.from_frame(lut)
        
        # set sklearn-standard alias
        self.n_features_in_ = self.n_nodes_in_
        
        # the number of nodes in the states must always match the number of nodes from 
        # the adjacency matrix
        if self.n_nodes_in_ != self.n_nodes_:
            raise ValueError(
                "State input must have the same number of nodes as A; "
                f"got {self.n_nodes_in_} nodes for "
                f"{self.n_nodes_} nodes in A"
            )
        
        # if user has provided separate node labels then overwrite the inferred ones
        node_labels = node_labels if node_labels is not None else node_labels_inferred
    
        # make sure that node labels are always convertable to index and have 
        # the expected length
        self.node_labels_ = None if node_labels is None else _coerce_labels(node_labels,self.n_nodes_in_,"node_labels")
        
    # TODO: Is it true that xr can be also array like?
    def fit(self, X=None,y=None,*,x0=None,xf=None,node_labels=None):
        """Validate and fit all inputs, i.e. check nct parameters and check state inputs"""
        
        self._fit_cache()

        # validate boolean options
        self.store_state_trajectories_ = _validate_boolean(self.store_state_trajectories, "store_state_trajectories")
        self.store_control_trajectories_ = _validate_boolean(self.store_control_trajectories,"store_control_trajectories")
        
        # validate all inputs used to compute transitions
        # TODO: Maybe this should already provided the fitted outputs?
        _validate_transition_inputs(
            self.A,
            self.T,
            self.B,
            self.rho,
            self.S,
            self.energy_type,
            self.system,
            self.xr,
            self.expm_version,
            self.normalize_A,
            self.c
        )
        
        # validate and fit matrices. Sets A_, B_, S_
        self._fit_matrices()
        
        # validate and fit state input. Sets X_
        self._fit_states(X,x0,xf,node_labels)
        
        # set all other parameters
        self.T_ = self.T
        self.energy_type_ = self.energy_type
        self.system_ = self.system
        self.xr_ = self.xr
        self.expm_version_ = self.expm_version
        self.normalize_A_ = self.normalize_A
        self.c_ = self.c
        
        if self.energy_type == "minimal":
            self.rho_ = 1.0
        else:
            self.rho_ = self.rho
        
        # FIXME: What should .fit() return?
        return self

    def _transform_states(self,X):
        """Return states as a 2D NumPy array suitable for transition computation."""
        if self.X_type_ == "tabular_like":
            return X.to_numpy()
    
        if self.X_type_ == "niimg_like":
            return self.masker_.transform(X)
    
    def transform(self,X=None,*,x0=None,xf=None,state_labels=None,order='permutations'):
        """Return integrated control energy with shape transitions by nodes.
        
        state_labels : list-like or MultiIndex-like, optional
            Labels for states. When supplied, :meth:`transform` returns a DataFrame
            whose row index identifies each transition endpoint. If omitted, 
            labels are inferred from input.
        order : {"combinations", "permutations", "product", "stability"}, \
            default="permutations"
            State-pair selection and ordering.
            
            """

        # check that all needed inputs exist
        check_is_fitted(
            self,
            attributes=[
                "A_norm_",
                "B_",
                "S_",
                "rho_",
                "X_type_",
                "n_nodes_in_",
                "node_labels_",
            ],
        )
    

        # Resolve transform input into a consistent representation.
        X_resolved = _resolve_state_input(X=X,x0=x0,xf=xf)
    
        # Input type must match what was seen during fit.
        X_type = _is_niimg_or_tabular_like(X_resolved)
    
        if X_type != self.X_type_:
            raise TypeError(
                "State input type must match the type used during fit; "
                f"fit received {self.X_type_!r}, but transform received "
                f"{X_type!r}"
            )
        
        # transform into (n_states, n_nodes).
        X = self._transform_states(X_resolved)
        n_states = X.shape[0]
        
        # Validate requested transition ordering.
        order = _validate_transition_order(n_states, order)
    
        # number of nodes must match what was seen during fit.
        if X.shape[1] != self.n_nodes_in_:
            raise ValueError(
                "State input must contain the same number of nodes as seen "
                f"during fit; expected {self.n_nodes_in_}, "
                f"got {X.shape[1]}"
            )
        
        # compute trajectories
        cached_transition = self._cache(get_transition_trajectories, func_memory_level=1)

        state_trajectories, control_trajectories, errors = cached_transition(
            A=self.A_norm_,
            T=self.T_,
            B=self.B_,
            X=X,
            rho=self.rho_,
            S=self.S_,
            order=order,
            system=self.system_,
            xr=self.xr_,
            expm_version=self.expm_version_,
        )
        
        # if set by user during init, store trajectories otherwise don't expose them
        if self.store_state_trajectories_:
            self.state_trajectories_ = state_trajectories
        else:
            self.__dict__.pop("state_trajectories_", None)
            
        if self.store_control_trajectories_:
            self.control_trajectories_ = control_trajectories
        else:
            self.__dict__.pop("control_trajectories_",None)
        
        self.errors_ = errors
        
        # integrate control inputs
        transition_energy = get_transition_energy(control_trajectories)
        
        # infer state labels
        if X_type == "tabular_like":
            state_labels_inferred = X_resolved.index
    
        elif X_type == "niimg_like":
            state_labels_inferred = None
    
        # if user has provided separate state labels then overwrite the inferred ones
        state_labels = state_labels if state_labels is not None else state_labels_inferred
        
        # make sure that state labels are always convertable to index and have the expected length
        state_labels = None if state_labels is None else _coerce_labels(state_labels,n_states,"state_labels")
        
        # compute transition labels
        transition_labels = None if state_labels is None else _state_transition_index(state_labels,order)

        # Return df that has either both columns and index, only columns, only index, no columns and no index
        df_transition_energy = pd.DataFrame(transition_energy,index=transition_labels,columns=self.node_labels_)
                
        return df_transition_energy

    def fit_transform(
        self,
        X=None,
        y=None,
        *,
        x0=None,
        xf=None,
        node_labels=None,
        state_labels=None,
        order="permutations",
    ):
        """Fit and transform states supplied as ``X`` or as ``x0`` and ``xf``."""
        return self.fit(
            X,
            y,
            x0=x0,
            xf=xf,
            node_labels=node_labels,
        ).transform(
            X,
            x0=x0,
            xf=xf,
            state_labels=state_labels,
            order=order,
        )
            
    def get_errors(self):
        """Return numerical errors from the most recent transform call."""
        check_is_fitted(self, attributes=["errors_"])
        return self.errors_.copy()

    # TODO: Should be split up into .get_state_trajectories and .get_control_trajectories
    # TODO: Should also output pandas dataframes with shape (time,nodes,transition)
    def get_transition_arrays(self):
        """Return retained trajectory arrays, or ``None`` when disabled."""

        state_trajectories = getattr(self, "state_trajectories_", None)
        control_trajectories = getattr(self, "control_trajectories_", None)
        
        return (
            None if state_trajectories is None else state_trajectories.copy(),
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
            names = _coerce_labels(input_features, self.n_features_in_, "input_features")
        elif self.node_labels_ is not None:
            names = self.node_labels_
        else:
            names = [f"node_{index}" for index in range(self.n_features_in_)]
        result = np.empty(self.n_features_in_, dtype=object)
        result[:] = list(names)
        return result

__all__ = [
    "Transitioner",
    "get_transition_trajectories",
    "get_transition_energy",
]