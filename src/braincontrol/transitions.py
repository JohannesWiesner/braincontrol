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
    _set_transition_order,
    _state_transition_index,
)
from braincontrol.utils.validation import (
    _is_niimg_or_tabular_like,
    _resolve_array_or_identity,
    _resolve_state_input,
    _validate_adjacency_inputs,
    _validate_boolean,
    _validate_choice,
    _validate_same_shape,
    _validate_square_matrix_or_identity,
    _validate_transition_order,
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
    xr : array-like or str, default="zero"
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
    
    def _fit_nct_parameters(self,
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
        c,
    ):
        """Validate and resolve Network Control Theory parameters"""
        
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
    
        _validate_adjacency_inputs(
            A,
            normalize_A,
            c,
        )
    
        _validate_square_matrix_or_identity(
            B,
            "B",
        )
    
        if energy_type == "minimal":
            if rho is not None or S is not None:
                raise ValueError(
                    "rho and S must both be None when "
                    "energy_type='minimal'"
                )
    
            rho = 1.0
    
        else:
            if rho is None or S is None:
                raise ValueError(
                    "rho and S must both be provided when "
                    "energy_type='optimal'"
                )
    
            if (
                isinstance(rho, (bool, np.bool_))
                or not isinstance(rho, (float, np.floating))
                or not np.isfinite(rho)
                or rho <= 0.0
                or rho > 1.0
            ):
                raise ValueError(
                    "rho must be a positive finite float between 0 and 1 when "
                    "energy_type='optimal"
                )
    
            _validate_square_matrix_or_identity(
                S,
                "S",
            )
    
        if isinstance(T, (bool, np.bool_)) or not np.isscalar(T):
            raise TypeError(
                "T must be a scalar number"
            )
    
        if not np.isfinite(T) or T <= 0:
            raise ValueError(
                "T must be a positive finite number"
            )
    
        if system == "discrete":
            if not isinstance(T, (int, np.integer)) or T < 2:
                raise ValueError(
                    "T must be an integer of at least 2 "
                    "for a discrete system"
                )
    
        elif not isinstance(T, (float, np.floating)):
            raise TypeError(
                "T must be a float for a continuous system"
            )
    
        # Resolve matrices. Everything is checked against A from here on
        A = np.asarray(A)
        n_nodes = A.shape[0]
    
        A_norm = (
            A.copy()
            if not normalize_A
            else matrix_normalization(
                A,
                system,
                c,
            )
        )
    
        B = _resolve_array_or_identity(
            B,
            n_nodes,
        )
    
        if energy_type == "minimal":
            S = np.zeros_like(A)
        else:
            S = _resolve_array_or_identity(
                S,
                n_nodes,
            )
    
        _validate_same_shape(
            [A, B, S],
            ["A", "B", "S"],
        )
    
        return {
            "A": A,
            "A_norm": A_norm,
            "B": B,
            "S": S,
            "T": T,
            "rho": rho,
            "energy_type": energy_type,
            "system": system,
            "xr": xr,
            "expm_version": expm_version,
            "normalize_A": normalize_A,
            "c": c,
            "n_nodes": n_nodes,
        }
        
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
            self.n_state_nodes_ = self.X_.shape[1]
            node_labels_inferred = self.X_.columns
    
        elif self.X_type_ == "niimg_like":
            
            self._fit_masker(self.X_)
            self.n_state_nodes_ = self.masker_.n_elements_
            self.n_states_in_ = self.X_.shape[3]

            lut = self.masker_.lut_
            lut = lut.loc[lut["index"] != self.masker_.background_label].reset_index(drop=True)
            node_labels_inferred = pd.MultiIndex.from_frame(lut)
        
        # set sklearn-standard alias
        self.n_features_in_ = self.n_state_nodes_
        
        # the number of nodes in the states must always match the number of nodes from 
        # the adjacency matrix
        if self.n_state_nodes_ != self.n_nodes_:
            raise ValueError(
                "State input must have the same number of nodes as A; "
                f"got {self.n_state_nodes_} nodes for "
                f"{self.n_nodes_} nodes in A"
            )
        
        # if user has provided separate node labels then overwrite the inferred ones
        node_labels = node_labels if node_labels is not None else node_labels_inferred
    
        # make sure that node labels are always convertable to index and have 
        # the expected length
        self.node_labels_ = None if node_labels is None else _coerce_labels(node_labels,self.n_state_nodes_,"node_labels")
        
    # TODO: Is it true that xr can be also array like?
    def fit(self, X=None,y=None,*,x0=None,xf=None,node_labels=None):
        """Validate and fit all inputs, i.e. check nct parameters and check state inputs"""
        
        self._fit_cache()
        
        # validate all inputs used to compute transitions
        nct_parameters = self._fit_nct_parameters(
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
            self.c,
        )
        
        for name, value in nct_parameters.items():
            setattr(self, f"{name}_", value)

        # validate and fit state input. Sets X_
        self._fit_states(X,x0,xf,node_labels)
        
        # validate boolean options
        self.store_state_trajectories_ = _validate_boolean(self.store_state_trajectories, "store_state_trajectories")
        self.store_control_trajectories_ = _validate_boolean(self.store_control_trajectories,"store_control_trajectories")
        
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
        if X.shape[1] != self.n_state_nodes_:
            raise ValueError(
                "State input must contain the same number of nodes as seen "
                f"during fit; expected {self.n_state_nodes_}, "
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