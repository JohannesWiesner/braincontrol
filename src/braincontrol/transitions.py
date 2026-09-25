"""
This module is the successor of :mod:`nict.single_subject`.  The functional
API accepts state matrices directly, while :class:`Transitioner`
also accepts image-like inputs through a scikit-learn compatible masker (for
example, :class:`nilearn.maskers.NiftiLabelsMasker`).
"""

import numpy as np

from braincontrol.utils.io import (
    _coerce_labels,
    _get_trajectory_array,
)

from braincontrol.utils.validation import (
    _validate_A,
    _resolve_A,
    _validate_B,
    _resolve_B,
    _validate_S,
    _resolve_S,
    _validate_positive_real,
    _validate_boolean,
    _validate_choice,
    _validate_same_shape,
    _validate_time_horizon,
    _validate_rho,
    _resolve_rho,
    _validate_xr,
    _validate_transition_states,
    _is_niimg_like,
    _validate_node_counts,
    _validate_state_array,
    _get_node_labels,
    _validate_node_labels
)

from nctpy.energies import get_control_inputs, integrate_u
from nilearn._utils.cache_mixin import CacheMixin
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.utils.validation import check_is_fitted
from itertools import combinations, permutations, product
from nilearn.maskers import BaseMasker
from nctpy.utils import matrix_normalization

###############################################################################
## functions
###############################################################################

# TODO: Find more suitable names for order choices
# TODO: Should also work with separate X0 and Xf (assuming that they are later concatenated)
def _set_transition_order(n_states, order):
    """Return the number and indices of requested state transitions.

    Each transition tuple contains ``(transition, source, target)``.  The
    ordering matches the corresponding iterator in :mod:`itertools`.
    """
    
    if not isinstance(n_states, (int, np.integer)) or isinstance(n_states, bool):
        raise TypeError("n_states must be an integer")
    if n_states < 1:
        raise ValueError("n_states must be at least 1")
        
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
    xr="xf",
    expm_version="scipy",
):
    """Compute state and control trajectories from prevalidated inputs.

    Parameters
    ----------
    A : ndarray of shape (n_nodes, n_nodes)
        A validated, normalized adjacency matrix.
    X : ndarray of shape (n_states, n_nodes)
        Validated state matrix with one state per row.
    T : float or int
        Time horizon.
    B : ndarray of shape (n_nodes, n_nodes)
        Validated control input matrix.
    rho : float
        Mixing parameter used by :func:`nctpy.energies.get_control_inputs`.
    S : ndarray of shape (n_nodes, n_nodes)
        Validated state-trajectory constraint matrix.
    order : {"combinations", "permutations", "product", "stability"}
        State-pair selection and ordering.
    system : {"continuous", "discrete"}, default="continuous"
        Time system used for the control computation.
    xr : array-like or str, default="xf"
        Reference state passed to ``nctpy``.
    expm_version : {"scipy", "eig"}, default="scipy"
        Matrix-exponential implementation used by ``nctpy``.

    Returns
    -------
    state_trajectories : ndarray
        State trajectories with shape
        ``(n_state_time_points, n_nodes, n_transitions)``.
    control_trajectories : ndarray
        Control trajectories with shape
        ``(n_control_time_points, n_nodes, n_transitions)``.
    errors : ndarray of shape (n_transitions, 2)
        Numerical errors reported by ``nctpy`` for each transition.
    """
    
    # TODO: might be better if this would be done in _set_transition_order
    # so this function only receives the indices
    n_transitions, transition_indices = _set_transition_order(X.shape[0],order)
    n_nodes = X.shape[1]

    # TODO: This should be exposed by nctpy! 0.001 is hardcoded for now, but it would be better if we could import STEP from nctpy so we always use nctpy as origin 
    if system == "continuous":
        n_state_time_points = int(np.round(T / 0.001) + 1)
        n_control_time_points = n_state_time_points
    else:
        n_state_time_points = T + 1
        n_control_time_points = T

    state_trajectories = np.empty((n_state_time_points, n_nodes, n_transitions),dtype=float)
    control_trajectories = np.empty((n_control_time_points, n_nodes, n_transitions),dtype=float)
    errors = np.empty((n_transitions, 2),dtype=float)

    for transition, source, target in transition_indices:
        
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

        state_trajectories[:, :, transition] = state_trajectory
        control_trajectories[:, :, transition] = control_trajectory
        errors[transition] = error

    return state_trajectories, control_trajectories, errors

def get_transition_energy(control_trajectories):
    """Integrate control trajectories for every transition."""
    
    n_nodes, n_transitions = control_trajectories.shape[1:]
    energies = np.empty((n_transitions, n_nodes))
    
    for transition in range(n_transitions):
        energies[transition] = integrate_u(control_trajectories[:, :, transition])
        
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
        self.normalize_A_ = normalize_A
        self.c_ = c
        self.store_state_trajectories_ = store_state_trajectories
        self.store_control_trajectories_ = store_control_trajectories
    
    def _fit_matrices(
        self,
        A,
        B,
        S,
    ):
        """Validate matrix inputs and fit matrix-related metadata."""
    
        # Validate matrices.
        _validate_A(A)
        _validate_B(B)
        _validate_S(S, self.energy_type_)
    
        # Extract node labels before resolving matrices.
        self.node_labels_["A"] = _get_node_labels(A)
        self.node_labels_["B"] = _get_node_labels(B)
        self.node_labels_["S"] = _get_node_labels(S)
    
        # A determines the number of nodes used to resolve B and S if they are `identity`
        n_nodes = A.shape[0]
    
        # Resolve matrices.
        A_resolved = _resolve_A(A)
        B_resolved = _resolve_B(B, n_nodes)
        S_resolved = _resolve_S(S,self.energy_type_,n_nodes)
    
        # Extract node counts after resolving
        self.n_nodes_["A"] = A_resolved.shape[0]
        self.n_nodes_["B"] = B_resolved.shape[0]
        self.n_nodes_["S"] = S_resolved.shape[0]
    
        # Validate resolved matrix shapes.
        _validate_same_shape(
            [A_resolved, B_resolved, S_resolved],
            ["A", "B", "S"],
        )
    
    def _fit_transform_masker(self, imgs):
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
    
        # TODO: Is this really needed?
        if not hasattr(masker, "n_elements_"):
            raise TypeError(
                "masker must expose n_elements_ after fitting."
            )
    
        return masker.transform(imgs)
    
    def _fit_states(
        self,
        X,
        X0,
        Xf,
        xr,
    ):
        """Validate state inputs and fit state-related metadata."""
    
        # Validate state-input configuration.
        _validate_transition_states(
            X,
            X0,
            Xf,
        )
        _validate_xr(
            xr,
            self.energy_type_,
        )
    
        for name, state_object in zip(
            ["X", "X0", "Xf", "xr"],
            [X, X0, Xf, xr],
        ):
            # Convert Niimg-like inputs to their state representation.
            if _is_niimg_like(state_object):
                state_object = self._fit_transform_masker(
                    state_object
                )
    
            # None and string inputs do not themselves define nodes.
            if state_object is None or isinstance(state_object, str):
                n_nodes = None
    
            else:
                _validate_state_array(
                    state_object,
                    name,
                )
    
                state_array = np.asarray(state_object)
    
                if state_array.ndim == 1:
                    n_nodes = state_array.shape[0]
                else:
                    n_nodes = state_array.shape[1]
    
            # Store only schema metadata, not the empirical state object.
            self.n_nodes_[name] = n_nodes
            self.node_labels_[name] = _get_node_labels(state_object)
    
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
        xr : array-like, str, or None, default="xf"
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
            self.normalize_A,
            self.c,
            self.store_state_trajectories,
            self.store_control_trajectories,
        )
    
        # Initialize fitted schema metadata.
        self.n_nodes_ = {}
        self.node_labels_ = {}
    
        # Fit matrix schema.
        self._fit_matrices(
            A,
            B,
            S,
        )
    
        # Fit state schema.
        self._fit_states(
            X,
            X0,
            Xf,
            xr,
        )
    
        # Validate the complete node schema.
        _validate_node_counts(
            self.n_nodes_
        )
        _validate_node_labels(
            self.node_labels_
        )
    
        return self


    
    # def transform(
    #     self,
    #     X=None,
    #     *,
    #     X0=None,
    #     xf=None,
    #     xr_override=None,
    #     state_labels=None,
    #     order="permutations",
    # ):
    #     """Return integrated control energy with shape transitions by nodes.
        
    #     xr_override : {"zero", "x0", "xf", "midpoint"}, array-like, Series, \
    #             Niimg-like, or None, optional
    #         Empirical reference state for these transitions. ``None`` uses
    #         the instance reference configured during construction.
    #     state_labels : list-like or MultiIndex-like, optional
    #         Labels for states. When supplied, :meth:`transform` returns a DataFrame
    #         whose row index identifies each transition endpoint. If omitted, 
    #         labels are inferred from input.
    #     order : {"combinations", "permutations", "product", "stability"}, \
    #         default="permutations"
    #         State-pair selection and ordering.
            
    #         """

    #     # check that all needed inputs exist
    #     # FIXME: Check this (some can be dropped others have to be added?)
    #     check_is_fitted(
    #         self,
    #         attributes=[
    #             "A_norm_",
    #             "B_",
    #             "S_",
    #             "rho_",
    #             "X_type_",
    #             "n_state_nodes_",
    #             "node_labels_",
    #             "xr_",
    #         ],
    #     )
    
    #     xr_transform = (
    #         self.xr_
    #         if xr_override is None
    #         else xr_override
    #     )

    #     # Check the reference against the fitted energy configuration before
    #     # resolving its concrete state representation.
    #     _resolve_rho_and_S(
    #         self.rho_,
    #         self.S_,
    #         self.energy_type_,
    #         self.n_nodes_,
    #     )

    #     # Resolve transform input into a consistent representation.
    #     (
    #         X_resolved,
    #         X_type,
    #         xr_resolved,
    #         xr_type,
    #         node_labels_transform,
    #     ) = _resolve_state_input(
    #         X=X,
    #         X0=X0,
    #         xf=xf,
    #         xr=xr_transform,
    #     )
        
    #     # Transform into (n_states, n_nodes), regardless of the concrete
    #     # representation used during fit.
    #     X, node_labels_transform = self._transform_states(
    #         X_resolved,
    #         X_type,
    #         node_labels_transform,
    #     )
    #     n_states = X.shape[0]
        
    #     # Validate requested transition ordering.
    #     order = _validate_transition_order(n_states, order)
    
    #     # number of nodes must match what was seen during fit.
    #     if X.shape[1] != self.n_state_nodes_:
    #         raise ValueError(
    #             "State input must contain the same number of nodes as seen "
    #             f"during fit; expected {self.n_state_nodes_}, "
    #             f"got {X.shape[1]}"
    #         )

    #     if (
    #         self.node_labels_ is not None
    #         and node_labels_transform is not None
    #         and not self.node_labels_.equals(node_labels_transform)
    #     ):
    #         raise ValueError(
    #             "Transform node labels must exactly match the fitted "
    #             "node labels, including their order"
    #         )

    #     xr_transform, _ = self._get_reference_state(
    #         xr_resolved,
    #         xr_type,
    #         self.masker_,
    #         X.shape[1],
    #     )
        
    #     # compute trajectories
    #     cached_transition = self._cache(get_transition_trajectories, func_memory_level=1)

    #     state_trajectories, control_trajectories, errors = cached_transition(
    #         A=self.A_norm_,
    #         T=self.T_,
    #         B=self.B_,
    #         X=X,
    #         rho=self.rho_,
    #         S=self.S_,
    #         order=order,
    #         system=self.system_,
    #         # nctpy still requires an xr value when S is the zero matrix used
    #         # for minimal energy, although the value cannot affect the cost.
    #         xr="zero" if xr_transform is None else xr_transform, # FIXME: I don't think this is right
    #         expm_version=self.expm_version_,
    #     )
        
    #     # if set by user during init, store trajectories otherwise don't expose them
    #     if self.store_state_trajectories_:
    #         self.state_trajectories_ = state_trajectories
    #     else:
    #         self.__dict__.pop("state_trajectories_", None)
            
    #     if self.store_control_trajectories_:
    #         self.control_trajectories_ = control_trajectories
    #     else:
    #         self.__dict__.pop("control_trajectories_",None)
        
    #     self.errors_ = errors
        
    #     # integrate control inputs
    #     transition_energy = get_transition_energy(control_trajectories)
        
    #     # infer state labels
    #     if X_type == "tabular_like":
    #         state_labels_inferred = X_resolved.index
    
    #     elif X_type == "niimg_like":
    #         state_labels_inferred = None
    
    #     # if user has provided separate state labels then overwrite the inferred ones
    #     state_labels = state_labels if state_labels is not None else state_labels_inferred
        
    #     # make sure that state labels are always convertable to index and have the expected length
    #     state_labels = None if state_labels is None else _coerce_labels(state_labels,n_states,"state_labels")
        
    #     # compute transition labels and store them
    #     transition_labels = None if state_labels is None else _state_transition_index(state_labels,order)
    #     self.transition_labels_ = transition_labels

    #     # Return df that has either both columns and index, only columns, only index, no columns and no index
    #     df_transition_energy = pd.DataFrame(transition_energy,index=transition_labels,columns=self.node_labels_)
                
    #     return df_transition_energy

    # def fit_transform(
    #     self,
    #     X=None,
    #     y=None,
    #     *,
    #     X0=None,
    #     xf=None,
    #     xr_override=None,
    #     node_labels=None,
    #     state_labels=None,
    #     order="permutations",
    # ):
    #     """Fit and transform states supplied as ``X`` or as ``X0`` and ``xf``.

    #     ``xr_override`` is a transform-time empirical reference. ``None``
    #     uses the reference configured on the instance.
    #     """
        
    #     return self.fit(
    #         X,
    #         y,
    #         x0=x0,
    #         xf=xf,
    #         node_labels=node_labels,
    #     ).transform(
    #         X,
    #         x0=x0,
    #         xf=xf,
    #         xr_override=xr_override,
    #         state_labels=state_labels,
    #         order=order,
    #     )
            
    def get_errors(self):
        """Return numerical errors from the most recent transform call."""
        
        check_is_fitted(self, attributes=["errors_"])
        return self.errors_.copy()

    def get_state_trajectories(self):
        """Return retained state trajectories as a labelled xarray DataArray."""
        
        return _get_trajectory_array(
                    getattr(self, "state_trajectories_", None),
                    node_labels=self.node_labels_,
                    transition_labels=self.transition_labels_,
                    name="state_trajectories",
                    )

    def get_control_trajectories(self):
        """Return retained control trajectories as a labelled xarray DataArray."""
        
        return _get_trajectory_array(
                    getattr(self, "control_trajectories_", None),
                    node_labels=self.node_labels_,
                    transition_labels=self.transition_labels_,
                    name="control_trajectory",
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