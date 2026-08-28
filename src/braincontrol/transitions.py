"""
This module is the successor of :mod:`nict.single_subject`.  The functional
API accepts state matrices directly, while :class:`Transitioner`
also accepts image-like inputs through a scikit-learn compatible masker (for
example, :class:`nilearn.maskers.NiftiLabelsMasker`).
"""

import numpy as np
import pandas as pd

from braincontrol.utils.io import (
    _coerce_labels,
    _get_trajectory_array,
    _set_transition_order, # FIXME: I think this should belong here
    _state_transition_index,
)

from braincontrol.utils.validation import (
    _resolve_array_or_identity,
    _resolve_state_input,
    _validate_square_matrix,
    _validate_positive_real,
    _validate_boolean,
    _validate_choice,
    _validate_same_shape,
    _validate_square_matrix_or_identity,
    _validate_time_horizon,
    _validate_transition_order,
    _resolve_energy_type_parameters,
)

from nctpy.energies import get_control_inputs, integrate_u
from nctpy.utils import matrix_normalization
from nilearn._utils.cache_mixin import CacheMixin
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.utils.validation import check_is_fitted

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
    
    n_transitions, transition_indices = _set_transition_order(X.shape[0],order)
    n_nodes = X.shape[1]

    # TODO: This should be exposed by nctpy!
    # 0.001 is hardcoded for now, but it would be better if we could
    # import STEP from nctpy so we always use nctpy as origin 
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
        with ``rho``, ``S``, and ``xr``, so all three must be ``None``.
        ``"optimal"`` requires all three parameters to be provided.
    xr : {"zero", "x0", "xf", "midpoint"}, array-like, Series, \
            Niimg-like, or None, default="xf"
        Default trajectory reference state. Optimal control requires a
        non-``None`` reference; minimal control requires ``None``. A compatible
        reference supplied as ``xr_override`` to :meth:`transform` overrides
        this value for that call.
    system : {"continuous", "discrete"}, default="continuous"
        Time system used for adjacency normalization and control computation.
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
        xr="xf",
        system="continuous",
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
        self.xr = xr
        self.system = system
        self.expm_version = expm_version
        self.masker = masker
        self.memory = memory
        self.memory_level = memory_level
        self.verbose = verbose
        self.store_state_trajectories = store_state_trajectories
        self.store_control_trajectories = store_control_trajectories
    
    def _fit_nct_parameters(
        self,
        A,
        T,
        B,
        rho,
        S,
        energy_type,
        system,
        expm_version,
        normalize_A,
        c,
        xr,
    ):
        """Validate and resolve Network Control Theory parameters."""
        
        # validate categorical parameters
        _validate_choice(energy_type,"energy_type",("minimal", "optimal"))
        _validate_choice(system,"system",("continuous", "discrete"))
        _validate_choice(expm_version,"expm_version",("scipy", "eig"))
    
        # validate adjacency matrix and normalization parameters
        _validate_square_matrix(A,"A")
        _validate_boolean(normalize_A,"normalize_A")
        _validate_positive_real(c,"c")
        
        # resolve adjacency matrix
        # TODO: I am not sure about this, but wouldn't it make sense to 
        # also to get the node_labels here in case A has them? Then, from 
        # here on every other matrix or state input that also has node labels must match the labels of A
        A = np.asarray(A)
        n_nodes = A.shape[0]

        if normalize_A == True:
            A_norm = matrix_normalization(A,system,c)
        elif normalize_A == False:
            A_norm = A.copy()
            
        # validate and resolve control input matrix B (depends on A)
        _validate_square_matrix_or_identity(B,"B")
        B = _resolve_array_or_identity(B,n_nodes)
        
        # resolve rho and S (depends on energy type and on A)
        rho, S = _resolve_energy_type_parameters(
            rho,
            S,
            energy_type,
            n_nodes,
            xr,
        )
        
        # validate time horizon
        _validate_time_horizon(T,system)
            
        # all matrices must have the same shape
        _validate_same_shape([A, B, S],["A", "B", "S"])
    
        return {
            "A_": A,
            "A_norm_": A_norm,
            "B_": B,
            "S_": S,
            "T_": T,
            "rho_": rho,
            "energy_type_": energy_type,
            "system_": system,
            "expm_version_": expm_version,
            "normalize_A_": normalize_A,
            "c_": c,
            "n_nodes_": n_nodes,
        }
    
    def _fit_masker(self, X, input_name="Image-like state input"):
        """Clone and fit the masker for Niimg-like state input."""
        
        if self.masker is None:
            raise ValueError(
                f"{input_name} requires a masker, for example "
                "NiftiLabelsMasker(labels_img=atlas)"
            )
    
        masker_fitted = clone(self.masker)
        masker_fitted.fit(X)
    
        required_attributes = ("n_elements_","lut_")
    
        missing_attributes = [
            attribute
            for attribute in required_attributes
            if not hasattr(masker_fitted, attribute)
        ]
    
        if missing_attributes:
            raise TypeError(
                "masker must expose the fitted attributes "
                f"{', '.join(required_attributes)}"
            )
    
        return masker_fitted

    @staticmethod
    def _get_masker_node_labels(masker):
        """Return node labels exposed by a fitted masker."""

        lut = masker.lut_.loc[
            masker.lut_["index"] != masker.background_label
        ].reset_index(drop=True)
        return pd.MultiIndex.from_frame(lut)
    
    def _get_reference_state(
        self,
        xr_resolved,
        xr_type,
        masker,
        n_state_nodes,
    ):
        """Convert a resolved reference state to the fitted node space."""

        if xr_type == "niimg_like":
            if masker is None:
                masker = self._fit_masker(
                    xr_resolved,
                    input_name="Image-like xr",
                )

            xr_array = np.asarray(masker.transform(xr_resolved))
            if xr_array.ndim == 2 and xr_array.shape[0] == 1:
                xr_array = xr_array[0]
            elif xr_array.ndim != 1:
                raise ValueError(
                    "Image-like xr must resolve to exactly one state; "
                    f"got transformed shape {xr_array.shape}"
                )

            if not np.all(np.isfinite(xr_array)):
                raise ValueError("Masked xr must contain only finite values")

            xr_fitted = xr_array.reshape(-1, 1)

        elif xr_type == "tabular_like":
            xr_fitted = np.asarray(xr_resolved).reshape(-1, 1).copy()

        else:
            xr_fitted = xr_resolved

        if (
            isinstance(xr_fitted, np.ndarray)
            and xr_fitted.shape[0] != n_state_nodes
        ):
            raise ValueError(
                "xr must have the same number of nodes as the state input; "
                f"got {xr_fitted.shape[0]} nodes for "
                f"{n_state_nodes} state nodes"
            )

        return xr_fitted, masker

    def _fit_states(
        self,
        X,
        x0,
        xf,
        xr,
        node_labels,
    ):
        """Resolve state input and fit state-related metadata."""

        (
            X_resolved,
            X_type,
            xr_resolved,
            xr_type,
            node_labels_inferred,
        ) = _resolve_state_input(X=X, x0=x0, xf=xf, xr=xr)

        if X_type == "tabular_like":

            masker_fitted = None

            n_states = X_resolved.shape[0]
            n_state_nodes = X_resolved.shape[1]

        elif X_type == "niimg_like":

            masker_fitted = self._fit_masker(X_resolved)

            n_states = X_resolved.shape[3]
            n_state_nodes = masker_fitted.n_elements_

            node_labels_inferred = self._get_masker_node_labels(
                masker_fitted
            )

        xr_fitted, masker_fitted = self._get_reference_state(
            xr_resolved,
            xr_type,
            masker_fitted,
            n_state_nodes,
        )
    
        # number of state nodes must match number of adjacency nodes.
        if n_state_nodes != self.n_nodes_:
            raise ValueError(
                "State input must have the same number of nodes as A; "
                f"got {n_state_nodes} nodes for "
                f"{self.n_nodes_} nodes in A"
            )
    
        # Explicit node labels override inferred labels.
        if node_labels is not None:
            node_labels_fitted = _coerce_labels(
                node_labels,
                n_state_nodes,
                "node_labels",
            )
    
        elif node_labels_inferred is not None:
            node_labels_fitted = _coerce_labels(
                node_labels_inferred,
                n_state_nodes,
                "node_labels",
            )
    
        else:
            node_labels_fitted = None
    
        # Store fitted state metadata.
        self.masker_ = masker_fitted
        self.X_type_ = X_type
        self.n_states_in_ = n_states
        self.n_state_nodes_ = n_state_nodes
        self.n_features_in_ = n_state_nodes
        self.node_labels_ = node_labels_fitted
        self.xr_ = xr_fitted
        
    def fit(
        self,
        X=None,
        y=None,
        *,
        x0=None,
        xf=None,
        node_labels=None,
    ):
        """Validate and fit the control model and state inputs."""
        
        self._fit_cache()
        
        # validate all nctpy inputs used to compute transitions
        nct_parameters = self._fit_nct_parameters(
            self.A,
            self.T,
            self.B,
            self.rho,
            self.S,
            self.energy_type,
            self.system,
            self.expm_version,
            self.normalize_A,
            self.c,
            self.xr,
        )
        
        for attr_,value in nct_parameters.items():
            setattr(self,f"{attr_}",value)
            
        # validate boolean options
        _validate_boolean(self.store_state_trajectories, "store_state_trajectories")
        self.store_state_trajectories_ = self.store_state_trajectories
        
        _validate_boolean(self.store_control_trajectories,"store_control_trajectories")
        self.store_control_trajectories_ = self.store_control_trajectories

        # validate and fit state input. Sets X_
        self._fit_states(X, x0, xf, self.xr, node_labels)

        # FIXME: What should .fit() return?
        return self

    def _transform_states(self, X, X_type, node_labels):
        """Map states and their labels into the fitted node space."""

        if X_type == "tabular_like":
            return X.to_numpy(), node_labels

        if self.masker_ is None:
            raise ValueError(
                "Image-like transform input requires a masker fitted "
                "during fit"
            )

        return (
            self.masker_.transform(X),
            self._get_masker_node_labels(self.masker_),
        )
    
    def transform(
        self,
        X=None,
        *,
        x0=None,
        xf=None,
        xr_override=None,
        state_labels=None,
        order="permutations",
    ):
        """Return integrated control energy with shape transitions by nodes.
        
        xr_override : {"zero", "x0", "xf", "midpoint"}, array-like, Series, \
                Niimg-like, or None, optional
            Empirical reference state for these transitions. ``None`` uses
            the instance reference configured during construction.
        state_labels : list-like or MultiIndex-like, optional
            Labels for states. When supplied, :meth:`transform` returns a DataFrame
            whose row index identifies each transition endpoint. If omitted, 
            labels are inferred from input.
        order : {"combinations", "permutations", "product", "stability"}, \
            default="permutations"
            State-pair selection and ordering.
            
            """

        # check that all needed inputs exist
        # FIXME: Check this (some can be dropped others have to be added?)
        check_is_fitted(
            self,
            attributes=[
                "A_norm_",
                "B_",
                "S_",
                "rho_",
                "X_type_",
                "n_state_nodes_",
                "node_labels_",
                "xr_",
            ],
        )
    
        xr_transform = (
            self.xr
            if xr_override is None
            else xr_override
        )

        # Check the reference against the fitted energy configuration before
        # resolving its concrete state representation.
        _resolve_energy_type_parameters(
            self.rho,
            self.S,
            self.energy_type_,
            self.n_nodes_,
            xr_transform,
        )

        # Resolve transform input into a consistent representation.
        (
            X_resolved,
            X_type,
            xr_resolved,
            xr_type,
            node_labels_transform,
        ) = _resolve_state_input(
            X=X,
            x0=x0,
            xf=xf,
            xr=xr_transform,
        )
        
        # Transform into (n_states, n_nodes), regardless of the concrete
        # representation used during fit.
        X, node_labels_transform = self._transform_states(
            X_resolved,
            X_type,
            node_labels_transform,
        )
        n_states = X.shape[0]
        
        # Validate requested transition ordering.
        order = _validate_transition_order(n_states, order)
    
        # number of nodes must match what was seen during fit.
        if X.shape[1] != self.n_state_nodes_:
            raise ValueError(
                "State input must contain the same number of nodes as seen "
                f"during fit; expected {self.n_state_nodes_}, "
                f"got {X.shape[1]}"
            )

        if (
            self.node_labels_ is not None
            and node_labels_transform is not None
            and not self.node_labels_.equals(node_labels_transform)
        ):
            raise ValueError(
                "Transform node labels must exactly match the fitted "
                "node labels, including their order"
            )

        xr_transform, _ = self._get_reference_state(
            xr_resolved,
            xr_type,
            self.masker_,
            X.shape[1],
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
            # nctpy still requires an xr value when S is the zero matrix used
            # for minimal energy, although the value cannot affect the cost.
            xr="zero" if xr_transform is None else xr_transform, # FIXME: I don't think this is right
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
        
        # compute transition labels and store them
        transition_labels = None if state_labels is None else _state_transition_index(state_labels,order)
        self.transition_labels_ = transition_labels

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
        xr_override=None,
        node_labels=None,
        state_labels=None,
        order="permutations",
    ):
        """Fit and transform states supplied as ``X`` or as ``x0`` and ``xf``.

        ``xr_override`` is a transform-time empirical reference. ``None``
        uses the reference configured on the instance.
        """
        
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
            xr_override=xr_override,
            state_labels=state_labels,
            order=order,
        )
            
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
