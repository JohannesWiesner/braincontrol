"""
This module is the successor of :mod:`nict.single_subject`.  The functional
API accepts state matrices directly, while :class:`Transitioner`
also accepts image-like inputs through a scikit-learn compatible masker (for
example, :class:`nilearn.maskers.NiftiLabelsMasker`).
"""

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
    _validate_transition_strategy
    )

from braincontrol.utils.validation.schema import (
    _validate_node_counts,
    _get_node_labels,
    _validate_node_labels,
    _get_common_node_labels,
    _validate_transform_node_labels,
    _get_node_count,
    _get_common_node_count,
    _get_state_labels,
    _validate_state_labels,
    _get_transition_labels,
)

from braincontrol.utils.processing.matrices import (
    _get_A,
    _get_B,
    _get_S,
    )

from braincontrol.utils.processing.transitions import (
    _get_transition_indices,
    _get_transition_trajectories,
    _get_transition_energy,
    _get_trajectory_array
    )

from braincontrol.utils.processing.parameters import _get_rho
from braincontrol.utils.processing.states import _get_xr

import numpy as np
import pandas as pd
from nilearn._utils.cache_mixin import CacheMixin
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.utils.validation import check_is_fitted
from nilearn.maskers import BaseMasker
from nctpy.utils import matrix_normalization

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
    
    # TODO: Outsource code as process_parameters for example processing.py and import it?
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
        rho = _get_rho(rho, energy_type)
    
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
    
    # TODO: Outsource to for example processing.py and import it?
    # TODO: Add more extensive docstring
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
        A = _get_A(A)
        B = _get_B(B, n_nodes)
        S = _get_S(S,self.energy_type_,n_nodes)
    
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
    
    # TODO: Outsource to for example processing/states.py and import it?
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
    
    # TODO: Outsource to for example processing/states.py and import it?
    def _process_states(self, states):
        """Process state inputs and extract their schema metadata.
    
        Each state input is converted to the representation used internally by
        the estimator while preserving available node and state labels as
        separate metadata.
    
        Niimg-like inputs are first transformed into node-level arrays using
        :meth:`_process_niimg`. Array-like inputs are validated and converted
        to NumPy arrays. ``None`` and string inputs are retained unchanged,
        because they may represent omitted or symbolic state inputs such as
        the reference state ``xr``.
    
        Node and state labels are extracted before conversion to NumPy arrays
        so that metadata provided by pandas objects is preserved.
    
        Parameters
        ----------
        states : dict
            Mapping from state-input names to their values. Typically contains
            the keys ``"X"``, ``"X0"``, ``"Xf"``, and ``"xr"``. Values may be
            array-like, Niimg-like, strings, or ``None``.
    
        Returns
        -------
        states_processed : dict
            Processed state inputs. Array-like and Niimg-like inputs are
            represented as NumPy arrays. Strings and ``None`` are retained
            unchanged.
    
        node_counts : dict
            Number of nodes represented by each processed state input.
            Entries are ``None`` for string or ``None`` inputs.
    
        node_labels : dict
            Node labels extracted from each state input when available.
            Entries are ``None`` when node labels cannot be inferred.
    
        state_labels : dict
            State labels extracted from each state input when available.
            Entries are ``None`` when state labels cannot be inferred.
    
        Raises
        ------
        TypeError
            If a state input is not a supported array-like object, or if a
            Niimg-like input is provided without a compatible masker.
    
        ValueError
            If an array-like state input has an invalid dimensionality or
            contains non-finite values.
        """
    
        states_processed = {}
        node_counts = {}
        node_labels = {}
        state_labels = {}
    
        for name, state in states.items():
    
            if _is_niimg_like(state):
                state = self._process_niimg(state)
    
            if state is None or isinstance(state, str):
                states_processed[name] = state
                node_counts[name] = None
                node_labels[name] = None
                state_labels[name] = None
                continue
    
            _validate_state_array(state, name)
    
            # Extract labels before converting to a NumPy array.
            node_labels[name] = _get_node_labels(state)
            state_labels[name] = _get_state_labels(state)
    
            state_processed = np.asarray(state)
    
            states_processed[name] = state_processed
            node_counts[name] = _get_node_count(state_processed)
    
        return (
            states_processed,
            node_counts,
            node_labels,
            state_labels,
        )
    
    # TODO: Add more extensive docstring
    def _fit_states(
        self,
        X,
        X0,
        Xf,
        xr,
    ):
        """Validate state inputs and return state-related schema metadata."""
    
        _validate_transition_states(
            X,
            X0,
            Xf,
        )
        _validate_xr(
            xr,
            self.energy_type_,
        )
    
        states = {
            "X": X,
            "X0": X0,
            "Xf": Xf,
            "xr": xr,
        }
    
        (
            states,
            node_counts,
            node_labels,
            _,
        ) = self._process_states(states)
    
        _validate_transition_strategy(
            self.transitions_,
            states["X"],
            states["X0"],
            states["Xf"],
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

    # TODO: Add more extensive docstring
    def _transform_states(
        self,
        X,
        X0,
        Xf,
        xr,
    ):
        """Validate and transform state inputs for NCT computation."""
    
        _validate_transition_states(
            X,
            X0,
            Xf,
        )
        _validate_xr(
            xr,
            self.energy_type_,
        )
    
        states = {
            "X": X,
            "X0": X0,
            "Xf": Xf,
            "xr": xr,
        }
    
        (
            states,
            node_counts,
            node_labels,
            state_labels,
        ) = self._process_states(states)
    
        _validate_transition_strategy(
            self.transitions_,
            states["X"],
            states["X0"],
            states["Xf"],
        )
    
        _validate_state_labels(
            state_labels
        )
    
        X = states["X"]
        X0 = states["X0"]
        Xf = states["Xf"]
        xr = _get_xr(
            states["xr"],
            self.energy_type_,
            )
    
        # FIXME: The following is too long! What is this even doing?
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
            
        # FIXME: This should be done in _validate_xr. As xr can be 
        # a niimg-like input by user this should be done after process states?
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
            state_labels,
        )
    
    # TODO: Add more extensive docstring
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
        """Validate transform-time node schema against the fitted schema.
    
        Parameters
        ----------
        node_counts : dict
            Number of nodes inferred from each transform input.
        node_labels : dict
            Node labels inferred from each transform input.
    
        Raises
        ------
        ValueError
            If the transform-time node count or node labels are incompatible
            with the schema established during fitting.
        """
    
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

    # TODO: Add more extensive docstring
    def transform(
        self,
        A,
        B="identity",
        S="identity",
        xr="xf",
        X=None,
        X0=None,
        Xf=None
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
            state_labels,
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
    
        self.transition_labels_ = _get_transition_labels(
            state_labels,
            transition_indices,
            n_initial_states=n_initial_states,
        )
    
        # Compute state and control trajectories.
        (
            state_trajectories,
            control_trajectories,
            errors,
        ) = _get_transition_trajectories(
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
        transition_energy = _get_transition_energy(
            control_trajectories
        )
    
        # Preserve available transition and node labels.
        if (
            self.transition_labels_ is not None
            or self.node_labels_ is not None
        ):
            transition_energy = pd.DataFrame(
                transition_energy,
                index=self.transition_labels_,
                columns=self.node_labels_,
            )
        
        return transition_energy
    
    def fit_transform(
        self,
        A,
        B="identity",
        S="identity",
        X=None,
        X0=None,
        Xf=None,
        xr="xf",
    ):
        """Fit the transformer and compute control energy for state transitions."""
    
        self.fit(
            A=A,
            B=B,
            S=S,
            X=X,
            X0=X0,
            Xf=Xf,
            xr=xr,
        )
    
        return self.transform(
            A=A,
            B=B,
            S=S,
            X=X,
            X0=X0,
            Xf=Xf,
            xr=xr,
        )
    
    def get_errors(self):
        """Return numerical errors from the most recent transform call."""
        
        check_is_fitted(self, attributes=["errors_"])
        return self.errors_.copy()

    def get_state_trajectories(self):
        """Return retained state trajectories as a labelled xarray DataArray.
    
        Returns
        -------
        xarray.DataArray or None
            State trajectories from the most recent transform call, with
            dimensions ``("time", "node", "transition")``. Returns None if
            state trajectories were not retained.
    
        Notes
        -----
        State trajectories are retained only when
        ``store_state_trajectories=True``.
        """
        check_is_fitted(
            self,
            attributes=["transition_labels_"],
        )
    
        return _get_trajectory_array(
            getattr(self, "state_trajectories_", None),
            node_labels=self.node_labels_,
            transition_labels=self.transition_labels_,
            name="state_trajectories",
        )
    
    def get_control_trajectories(self):
        """Return retained control trajectories as a labelled xarray DataArray.
    
        Returns
        -------
        xarray.DataArray or None
            Control trajectories from the most recent transform call, with
            dimensions ``("time", "node", "transition")``. Returns None if
            control trajectories were not retained.
    
        Notes
        -----
        Control trajectories are retained only when
        ``store_control_trajectories=True``.
        """
        check_is_fitted(
            self,
            attributes=["transition_labels_"],
        )
    
        return _get_trajectory_array(
            getattr(self, "control_trajectories_", None),
            node_labels=self.node_labels_,
            transition_labels=self.transition_labels_,
            name="control_trajectories",
        )

__all__ = [
    "Transitioner"
]