"""
This module is the successor of :mod:`nict.single_subject`.  The functional
API accepts state matrices directly, while :class:`Transitioner`
also accepts image-like inputs through a scikit-learn compatible masker (for
example, :class:`nilearn.maskers.NiftiLabelsMasker`).
"""

from braincontrol.utils.io import (
    _coerce_labels,
    _get_trajectory_array,
)

from braincontrol.utils.validation.matrices import (
    _validate_A,
    _validate_B,
    _validate_S,
    _validate_same_shape
    )

from braincontrol.utils.validation.parameters import (
    _validate_boolean,
    _validate_choice,
    _validate_positive_real,
    _validate_time_horizon,
    _validate_rho
    )

from braincontrol.utils.validation.states import (
    _is_niimg_like,
    _validate_transition_states,
    _validate_xr,
    _validate_state_array,
    )

from braincontrol.utils.validation.node_objects import (
    _validate_transition_strategy,
    _validate_node_counts,
    _get_node_labels,
    _validate_node_labels,
    _get_common_node_labels,
    _validate_transform_node_labels,
    _get_node_count,
    _get_common_node_count,
    )

from braincontrol.utils.resolving import (
    _resolve_A,
    _resolve_B,
    _resolve_S,
    _resolve_rho
    )

import numpy as np
from nctpy.energies import get_control_inputs, integrate_u
from nilearn._utils.cache_mixin import CacheMixin
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.utils.validation import check_is_fitted
from itertools import combinations, permutations, product
from nilearn.maskers import BaseMasker
from nctpy.utils import matrix_normalization

###############################################################################
## Functions
###############################################################################

# TODO: This function is too long
def _get_transition_indices(
    n_states,
    transitions,
    *,
    n_initial_states=None,
):
    """Return indices for the requested state transitions.

    Parameters
    ----------
    n_states : int
        Total number of states.

    transitions : {
        "directed",
        "undirected",
        "directed_with_self",
        "self",
        "all_to_all",
        "paired",
    }
        Strategy used to select state transitions.

        When a single set of states ``X`` is provided:

        - ``"directed"`` computes every directed transition between
          distinct states.
        - ``"undirected"`` computes one transition for each pair of
          distinct states.
        - ``"directed_with_self"`` computes every directed transition,
          including self-transitions.
        - ``"self"`` computes only self-transitions.

        When separate initial and final state sets ``X0`` and ``Xf`` are
        provided:

        - ``"all_to_all"`` computes every transition from a state in
          ``X0`` to a state in ``Xf``.
        - ``"paired"`` computes transitions between corresponding states
          in ``X0`` and ``Xf``.

    n_initial_states : int or None, default=None
        Number of initial states when ``n_states`` represents concatenated
        ``X0`` and ``Xf``. If None, ``n_states`` is assumed to describe a
        single state set ``X``.

    Returns
    -------
    transition_indices : list of tuple
        Transition indices represented as
        ``(transition, source, target)``.
    """
    if not isinstance(n_states, (int, np.integer)) or isinstance(
        n_states,
        bool,
    ):
        raise TypeError(
            "n_states must be an integer."
        )

    if n_states < 1:
        raise ValueError(
            "n_states must be at least 1."
        )

    # Transitions within a single state set.
    # TODO: Validation not needed anymore here as we have _validate_transition_strategy now
    if n_initial_states is None:
        _validate_choice(
            transitions,
            "transitions",
            (
                "directed",
                "undirected",
                "directed_with_self",
                "self",
            ),
        )

        indices = range(n_states)

        if transitions == "directed":
            pairs = permutations(
                indices,
                r=2,
            )

        elif transitions == "undirected":
            pairs = combinations(
                indices,
                r=2,
            )

        elif transitions == "directed_with_self":
            pairs = product(
                indices,
                repeat=2,
            )

        else:
            pairs = (
                (index, index)
                for index in indices
            )

    # Transitions from initial to final state sets.
    else:
        if (
            not isinstance(n_initial_states, (int, np.integer))
            or isinstance(n_initial_states, bool)
        ):
            raise TypeError(
                "n_initial_states must be an integer."
            )

        if not 1 <= n_initial_states < n_states:
            raise ValueError(
                "n_initial_states must be at least 1 and smaller "
                "than n_states."
            )

        _validate_choice(
            transitions,
            "transitions",
            (
                "all_to_all",
                "paired",
            ),
        )

        n_final_states = n_states - n_initial_states

        initial_indices = range(
            n_initial_states
        )
        final_indices = range(
            n_initial_states,
            n_states,
        )

        if transitions == "all_to_all":
            pairs = product(
                initial_indices,
                final_indices,
            )

        else:
            if n_initial_states != n_final_states:
                raise ValueError(
                    "X0 and Xf must contain the same number of states "
                    "when transitions='paired'."
                )

            pairs = zip(
                initial_indices,
                final_indices,
            )

    return [
        (transition, source, target)
        for transition, (source, target) in enumerate(pairs)
    ]

# TODO (#26): Make more memory efficient by predefining empty arrays that have
# n_transitions x n_timepoints x n_nodes. 
# TODO (#26): Make more computaionally efficient by using parallelization. 
def get_transition_trajectories(
    A,
    X,
    transition_indices,
    T,
    B,
    rho,
    S,
    *,
    system="continuous",
    xr="xf",
    expm_version="scipy",
):
    """Compute state and control trajectories for state transitions.

    Parameters
    ----------
    A : ndarray of shape (n_nodes, n_nodes)
        Adjacency matrix.

    X : ndarray of shape (n_states, n_nodes)
        State matrix.

    transition_indices : list of tuple
        State transitions represented as
        ``(transition, source, target)``.

    T : float or int
        Time horizon.

    B : ndarray of shape (n_nodes, n_nodes)
        Control input matrix.

    rho : float
        State-trajectory constraint parameter.

    S : ndarray of shape (n_nodes, n_nodes)
        State-trajectory constraint matrix.

    system : {"continuous", "discrete"}, default="continuous"
        System dynamics.

    xr : {"xf", "zero"} or ndarray, default="xf"
        Reference state.

    expm_version : {"scipy", "eig"}, default="scipy"
        Matrix exponential implementation.

    Returns
    -------
    state_trajectories : list of ndarray
        State trajectory for each transition.

    control_trajectories : list of ndarray
        Control trajectory for each transition.

    errors : ndarray of shape (n_transitions, 2)
        Numerical errors for all transitions.
    """

    n_transitions = len(transition_indices)

    state_trajectories = []
    control_trajectories = []
    errors = np.zeros((n_transitions, 2))

    for transition, source, target in transition_indices:
        (
            state_trajectory,
            control_trajectory,
            error,
        ) = get_control_inputs(
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
        errors[transition] = error

    return (
        state_trajectories,
        control_trajectories,
        errors,
    )

def get_transition_energy(control_trajectories):
    """Integrate control trajectories for every transition.

    Parameters
    ----------
    control_trajectories : list of ndarray
        Control trajectories for all transitions. Each element has shape
        (n_timepoints, n_nodes).

    Returns
    -------
    energies : ndarray of shape (n_transitions, n_nodes)
        Integrated control energy for each transition and node.
    """

    n_transitions = len(control_trajectories)
    n_nodes = control_trajectories[0].shape[1]

    energies = np.empty(
        (n_transitions, n_nodes)
    )

    for transition, control_trajectory in enumerate(
        control_trajectories
    ):
        energies[transition] = integrate_u(
            control_trajectory
        )

    return energies

###############################################################################
## Class
###############################################################################

class Transitioner(TransformerMixin, CacheMixin, BaseEstimator, auto_wrap_output_keys=None):
    """Transform state transitions into node-level control energies.
    
        ``Transitioner`` computes Network Control Theory (NCT) energies from an
        adjacency matrix and a set of states. State input can be provided either
        as ``X``, containing one state per row, or as separate ``X0`` and ``Xf``
        inputs.
    
        State inputs may also be Niimg-like objects when a compatible Nilearn
        masker is provided. The masker is fit and applied independently to each
        Niimg-like input to obtain its node-level state representation.
    
        :meth:`fit` validates the estimator configuration and empirical inputs and
        records their node schema. :meth:`transform` validates new empirical inputs
        against the fitted schema, resolves the inputs required for the NCT
        calculation, and computes transition energies.
    
        State and control trajectories from the most recent transform call can
        optionally be retained and accessed with :meth:`get_state_trajectories`
        and :meth:`get_control_trajectories`. Numerical errors reported by
        ``nctpy`` are available through :meth:`get_errors`.
    
        Parameters
        ----------
        T : float
            Positive time horizon. Discrete systems require an integer of at least
            two.
        normalize_A : bool, default=True
            If ``True``, normalize the adjacency matrix during :meth:`transform`
            using :func:`nctpy.utils.matrix_normalization` for the selected
            ``system``. If ``False``, use the adjacency matrix as provided.
        c : float, default=1
            Positive normalization constant passed to
            :func:`nctpy.utils.matrix_normalization`.
        energy_type : {"minimal", "optimal"}, default="optimal"
            Type of control energy to compute. Minimal energy requires ``rho``,
            ``S``, and ``xr`` to be omitted. Optimal energy requires these
            parameters to be specified.
        rho : float or None, default=1.0
            Positive mixing parameter for optimal control energy. Must be less
            than or equal to 1. For minimal control energy, ``rho`` must be
            ``None``.
        system : {"continuous", "discrete"}, default="continuous"
            Time system used for adjacency normalization and control computation.
        expm_version : {"scipy", "eig"}, default="scipy"
            Matrix-exponential implementation forwarded to ``nctpy``.
        transitions : {"directed", "undirected", "directed_with_self", "self", \
                       "all_to_all", "paired"}, default="directed"
            Strategy used to select state transitions. The available strategies
            depend on whether states are provided as X or as separate X0 and Xf
            inputs.
        masker : nilearn.maskers.BaseMasker or None, default=None
            Masker used to convert Niimg-like state inputs to node-level state
            representations. Required when Niimg-like state inputs are provided.
        memory : None, str, pathlib.Path, or joblib.Memory, default=None
            Cache location for state-transition computations. Caching is disabled
            when ``None``.
        memory_level : int, default=1
            Cache state-transition computations when this value is at least 1.
        verbose : int, default=0
            Verbosity forwarded to Nilearn's caching infrastructure.
        store_state_trajectories : bool, default=False
            Whether to retain state trajectories from the most recent transform
            call.
        store_control_trajectories : bool, default=False
            Whether to retain control trajectories from the most recent transform
            call.
        """

    def __init__(
        self,
        T,
        normalize_A=True,
        c=1,
        energy_type="optimal",
        rho=1.0,
        system="continuous",
        expm_version="scipy",
        transitions="directed",
        masker=None,
        memory=None,
        memory_level=1,
        verbose=0,
        store_state_trajectories=False,
        store_control_trajectories=False,
    ):
        self.T = T
        self.normalize_A = normalize_A
        self.c = c
        self.energy_type = energy_type
        self.rho = rho
        self.system = system
        self.expm_version = expm_version
        self.transitions = transitions
        self.masker = masker
        self.memory = memory
        self.memory_level = memory_level
        self.verbose = verbose
        self.store_state_trajectories = store_state_trajectories
        self.store_control_trajectories = store_control_trajectories
    
    def _fit_parameters(
        self,
        T,
        rho,
        energy_type,
        system,
        expm_version,
        transitions,
        normalize_A,
        c,
        store_state_trajectories,
        store_control_trajectories,
    ):
        """Validate and fit Network Control Theory parameters."""
    
        # Validate categorical parameters.
        _validate_choice(
            energy_type,
            "energy_type",
            ("minimal", "optimal"),
        )
        _validate_choice(
            system,
            "system",
            ("continuous", "discrete"),
        )
        _validate_choice(
            expm_version,
            "expm_version",
            ("scipy", "eig"),
        )
        _validate_choice(
            transitions,
            "transitions",
            (
                "directed",
                "undirected",
                "directed_with_self",
                "self",
                "all_to_all",
                "paired",
            ),
        )
    
        # Validate normalization parameters.
        _validate_boolean(normalize_A, "normalize_A")
        _validate_positive_real(c, "c")
    
        # Validate time horizon.
        _validate_time_horizon(T, system)
    
        # Validate and resolve rho.
        _validate_rho(rho, energy_type)
        rho = _resolve_rho(rho, energy_type)
    
        # Validate storage options.
        _validate_boolean(
            store_state_trajectories,
            "store_state_trajectories",
        )
        _validate_boolean(
            store_control_trajectories,
            "store_control_trajectories",
        )
    
        # Store fitted parameters.
        self.T_ = T
        self.rho_ = rho
        self.energy_type_ = energy_type
        self.system_ = system
        self.expm_version_ = expm_version
        self.transitions_ = transitions
        self.normalize_A_ = normalize_A
        self.c_ = c
        self.store_state_trajectories_ = store_state_trajectories
        self.store_control_trajectories_ = store_control_trajectories
    
    def _process_matrices(
        self,
        A,
        B,
        S,
    ):
        """Validate and resolve matrix inputs and extract their schema metadata."""
    
        # Validate matrices.
        _validate_A(A)
        _validate_B(B)
        _validate_S(S, self.energy_type_)
    
        # Extract node labels before resolving matrices.
        node_labels = {
            "A": _get_node_labels(A),
            "B": _get_node_labels(B),
            "S": _get_node_labels(S),
        }
    
        # A determines the number of nodes used to resolve B and S.
        n_nodes = A.shape[0]
    
        # Resolve matrices.
        A = _resolve_A(A)
        B = _resolve_B(B, n_nodes)
        S = _resolve_S(S,self.energy_type_,n_nodes)
    
        # Validate resolved matrix shapes.
        _validate_same_shape(
            [A, B, S],
            ["A", "B", "S"],
        )
    
        # Extract node counts from the resolved matrices.
        node_counts = {
            "A": A.shape[0],
            "B": B.shape[0],
            "S": S.shape[0],
        }
    
        return (
            A,
            B,
            S,
            node_counts,
            node_labels,
        )
        
    def _fit_matrices(
        self,
        A,
        B,
        S,
    ):
        """Validate matrix inputs and return matrix-related schema metadata."""
    
        (
            _,
            _,
            _,
            node_counts,
            node_labels,
        ) = self._process_matrices(
            A,
            B,
            S,
        )
    
        return node_counts, node_labels
    
    def _process_niimg(self, imgs):
        """Fit and apply a masker to Niimg-like input.
    
        Parameters
        ----------
        imgs : Niimg-like
            Image data to transform.
    
        Returns
        -------
        array-like
            Masked image data.
        """
        if not isinstance(self.masker, BaseMasker):
            raise TypeError(
                "masker must be a nilearn BaseMasker instance "
                "when Niimg-like state inputs are provided."
            )
    
        masker = clone(self.masker).fit(imgs)
    
        return masker.transform(imgs)
    
    def _process_states(self, states):
        """Transform state inputs into nctpy-digestible format and extract their schema metadata."""
    
        states_processed = {}
        node_counts = {}
        node_labels = {}
    
        for name, state in states.items():
    
            if _is_niimg_like(state):
                state = self._process_niimg(state)
    
            if state is None or isinstance(state, str):
                states_processed[name] = state
                node_counts[name] = None
                node_labels[name] = None
                continue
    
            # validate that object is 1D/2D array-ish
            _validate_state_array(state, name)
            state_processed = np.asarray(state)
            
            # get number of nodes from preprocesed state and node_labels from 
            # original input
            n_nodes = _get_node_count(state_processed)
            node_labels[name] = _get_node_labels(state)
    
            states_processed[name] = state_processed
            node_counts[name] = n_nodes
    
        return (
            states_processed,
            node_counts,
            node_labels,
        )
    
    def _fit_states(
        self,
        X,
        X0,
        Xf,
        xr,
    ):
        """Validate state inputs and return state-related schema metadata."""
    
        # Validate state-input configuration.
        _validate_transition_states(X, X0, Xf)
        _validate_transition_strategy(self.transitions_,X)
        _validate_xr(xr, self.energy_type_)
    
        states = {
            "X": X,
            "X0": X0,
            "Xf": Xf,
            "xr": xr,
        }
    
        _, node_counts, node_labels = self._process_states(
            states
        )
    
        return node_counts, node_labels
    
    def fit(
        self,
        A,
        B="identity",
        S="identity",
        X=None,
        X0=None,
        Xf=None,
        xr="xf"
    ):
        """Fit the Network Control Theory transformer.
    
        Parameters
        ----------
        A : array-like of shape (n_nodes, n_nodes)
            Adjacency matrix.
        B : array-like of shape (n_nodes, n_nodes) or "identity", \
                default="identity"
            Control input matrix.
        S : array-like of shape (n_nodes, n_nodes), "identity", or None, \
                default="identity"
            State-trajectory constraint matrix.
        X : array-like, Niimg-like, or None, default=None
            State input.
        X0 : array-like, Niimg-like, or None, default=None
            Initial-state input.
        Xf : array-like, Niimg-like, or None, default=None
            Final-state input.
        xr : array-like, Niimg-like, str, or None, default="xf"
            Reference state.
    
        Returns
        -------
        self : Transitioner
            Fitted estimator.
        """
        # Validate and store estimator parameters.
        self._fit_parameters(
            self.T,
            self.rho,
            self.energy_type,
            self.system,
            self.expm_version,
            self.transitions,
            self.normalize_A,
            self.c,
            self.store_state_trajectories,
            self.store_control_trajectories,
        )
    
        # Fit matrix schema.
        matrix_node_counts, matrix_node_labels = self._fit_matrices(
            A,
            B,
            S,
        )
    
        # Fit state schema.
        state_node_counts, state_node_labels = self._fit_states(
            X,
            X0,
            Xf,
            xr,
        )
    
        # Combine matrix and state schema metadata.
        node_counts = {
            **matrix_node_counts,
            **state_node_counts,
        }
    
        node_labels = {
            **matrix_node_labels,
            **state_node_labels,
        }
    
        # Validate the complete node schema.
        _validate_node_counts(node_counts)
        _validate_node_labels(node_labels)
    
        # Store the validated common node schema.
        self.n_nodes_ = _get_common_node_count(node_counts)
        self.node_labels_ = _get_common_node_labels(node_labels)
    
        return self

    def _transform_states(
        self,
        X,
        X0,
        Xf,
        xr,
    ):
        """Validate and transform state inputs for NCT computation."""
    
        # Validate state-input configuration.
        _validate_transition_states(X, X0, Xf)
        _validate_transition_strategy(self.transitions_,X,)
        _validate_xr(xr, self.energy_type_)
    
        states = {
            "X": X,
            "X0": X0,
            "Xf": Xf,
            "xr": xr,
        }
    
        # Process states and extract their schema metadata.
        states, node_counts, node_labels = self._process_states(
            states
        )
    
        X = states["X"]
        X0 = states["X0"]
        Xf = states["Xf"]
        xr = states["xr"]
    
        # TODO: the following is too long
        # Arrange transition states for NCT computation.
        if X is not None:
            if X.ndim == 1:
                X = X[np.newaxis, :]
    
            n_initial_states = None
    
        else:
            if X0.ndim == 1:
                X0 = X0[np.newaxis, :]
    
            if Xf.ndim == 1:
                Xf = Xf[np.newaxis, :]
    
            n_initial_states = X0.shape[0]
    
            X = np.concatenate(
                [X0, Xf],
                axis=0,
            )
    
        # Arrange the reference state for NCT computation.
        if xr is not None and not isinstance(xr, str):
            if xr.ndim == 2:
                if xr.shape[0] != 1:
                    raise ValueError(
                        "xr must describe a single reference state."
                    )
    
                xr = xr[0]
    
        return (
            X,
            xr,
            n_initial_states,
            node_counts,
            node_labels,
        )

    def _transform_matrices(
        self,
        A,
        B,
        S,
    ):
        """Validate and transform matrix inputs for NCT computation."""
    
        (
            A,
            B,
            S,
            node_counts,
            node_labels,
        ) = self._process_matrices(
            A,
            B,
            S,
        )
    
        # Normalize the adjacency matrix used for NCT.
        if self.normalize_A_:
            A = matrix_normalization(
                A,
                self.system_,
                self.c_,
            )
    
        return (
            A,
            B,
            S,
            node_counts,
            node_labels,
        )

    def _validate_transform_schema(
        self,
        node_counts,
        node_labels,
    ):
        """Validate transform-time node schema against the fitted schema."""
    
        # Extract the common transform-time node count.
        transform_n_nodes = _get_common_node_count(
            node_counts
        )
    
        if transform_n_nodes != self.n_nodes_:
            raise ValueError(
                f"Transform inputs contain {transform_n_nodes} nodes, "
                f"but the fitted schema contains {self.n_nodes_} nodes."
            )
    
        # Extract the common transform-time node labels.
        transform_node_labels = _get_common_node_labels(
            node_labels
        )
    
        _validate_transform_node_labels(
            self.node_labels_,
            transform_node_labels,
        )

    # TODO: Work on state_labels
    def transform(
        self,
        A,
        B="identity",
        S="identity",
        xr="xf",
        X=None,
        X0=None,
        Xf=None,
        state_labels=None,
    ):
        """Compute control energy for state transitions."""
    
        check_is_fitted(
            self,
            attributes=[
                "T_",
                "rho_",
                "energy_type_",
                "system_",
                "expm_version_",
                "transitions_",
                "normalize_A_",
                "c_",
                "n_nodes_",
                "node_labels_",
            ],
        )
    
        # Transform matrices.
        (
            A,
            B,
            S,
            matrix_node_counts,
            matrix_node_labels,
        ) = self._transform_matrices(
            A,
            B,
            S,
        )
    
        # Transform states.
        (
            X,
            xr,
            n_initial_states,
            state_node_counts,
            state_node_labels,
        ) = self._transform_states(
            X,
            X0,
            Xf,
            xr,
        )
    
        # Combine transform-time node metadata.
        node_counts = {
            **matrix_node_counts,
            **state_node_counts,
        }
    
        node_labels = {
            **matrix_node_labels,
            **state_node_labels,
        }
    
        # Validate consistency among transform inputs.
        _validate_node_counts(
            node_counts
        )
        _validate_node_labels(
            node_labels
        )
    
        # Validate transform inputs against the fitted schema.
        self._validate_transform_schema(
            node_counts,
            node_labels,
        )
    
        # Determine requested state transitions.
        transition_indices = _get_transition_indices(
            X.shape[0],
            self.transitions_,
            n_initial_states=n_initial_states,
        )
    
        # Compute state and control trajectories.
        (
            state_trajectories,
            control_trajectories,
            errors,
        ) = get_transition_trajectories(
            A,
            X,
            transition_indices,
            self.T_,
            B,
            self.rho_,
            S,
            system=self.system_,
            xr=xr,
            expm_version=self.expm_version_,
        )
    
        # Store numerical errors from the most recent transform.
        self.errors_ = errors
    
        # Store trajectories from the most recent transform if requested.
        if self.store_state_trajectories_:
            self.state_trajectories_ = state_trajectories
        else:
            self.__dict__.pop(
                "state_trajectories_",
                None,
            )
    
        if self.store_control_trajectories_:
            self.control_trajectories_ = control_trajectories
        else:
            self.__dict__.pop(
                "control_trajectories_",
                None,
            )
    
        # Compute node-level control energy.
        transition_energy = get_transition_energy(
            control_trajectories
        )
    
        return transition_energy
    
    def get_errors(self):
        """Return numerical errors from the most recent transform call."""
        
        check_is_fitted(self, attributes=["errors_"])
        return self.errors_.copy()

    # FIXME: We must make sure that node_labels is a list
    # Update: It is a list-like object now
    def get_state_trajectories(self):
        """Return retained state trajectories as a labelled xarray DataArray."""
        
        return _get_trajectory_array(
                    getattr(self, "state_trajectories_", None),
                    node_labels=self.node_labels_,
                    transition_labels=self.transition_labels_,
                    name="state_trajectories",
                    )

    # FIXME: We must make sure that node_labels is a list
    # Update it is a list-like object now
    def get_control_trajectories(self):
        """Return retained control trajectories as a labelled xarray DataArray."""
        
        return _get_trajectory_array(
                    getattr(self, "control_trajectories_", None),
                    node_labels=self.node_labels_,
                    transition_labels=self.transition_labels_,
                    name="control_trajectory",
                    )

    # TODO: Not sure if this works currently. 
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