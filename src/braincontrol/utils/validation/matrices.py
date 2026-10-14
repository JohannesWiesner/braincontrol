#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Validation helpers for matrices."""

import numpy as np
import pandas as pd


def _validate_square_matrix(value, name):
    """Validate a finite, numeric, symmetric square matrix.

    Accept a NumPy array or pandas DataFrame representing a square matrix.
    The matrix must be two-dimensional, contain only finite numeric or
    boolean values, and be symmetric within NumPy's default floating-point
    tolerances.

    For pandas DataFrames, row and column labels must match exactly,
    including their order.

    Parameters
    ----------
    value : numpy.ndarray or pandas.DataFrame
        Matrix to validate.
    name : str
        Name of the matrix, used in error messages.

    Raises
    ------
    TypeError
        If the input is not a NumPy array or pandas DataFrame, or if it
        contains non-numeric values.
    ValueError
        If the matrix is not two-dimensional, square, finite, or symmetric,
        or if its DataFrame row and column labels do not match.

    Notes
    -----
    Symmetry is evaluated using ``numpy.allclose``, which permits small
    numerical differences between corresponding entries in the upper
    and lower triangles.
    """
    if not isinstance(value, (np.ndarray, pd.DataFrame)):
        raise TypeError(
            f"{name} must be a NumPy array or pandas DataFrame."
        )

    array = np.asarray(value)

    if array.ndim != 2:
        raise ValueError(
            f"{name} must be two-dimensional; got shape {array.shape}."
        )

    if array.shape[0] != array.shape[1]:
        raise ValueError(
            f"{name} must be square; got shape {array.shape}."
        )

    if not (
        np.issubdtype(array.dtype, np.number)
        or np.issubdtype(array.dtype, np.bool_)
    ):
        raise TypeError(f"{name} must contain numeric values.")

    try:
        finite = np.isfinite(array)
    except TypeError as exc:
        raise TypeError(f"{name} must contain finite numeric values.") from exc
    if not np.all(finite):
        raise ValueError(f"{name} must contain only finite values.")

    # All matrices must be symmetric (upper and lower triangle must be the same)
    if not np.allclose(array, array.T):
        raise ValueError(
            f"{name} must be symmetric."
        )

    if isinstance(value, pd.DataFrame) and not value.index.equals(value.columns):
        raise ValueError(
            f"{name} must have matching index and column labels."
        )


def _validate_square_matrix_or_identity(value, name):
    """Validate a diagonal matrix or the ``'identity'`` sentinel.

    Accept the string ``'identity'`` or a finite, numeric, symmetric
    square NumPy array or pandas DataFrame. Explicit matrices must
    additionally be diagonal, meaning that every off-diagonal entry
    is exactly zero.

    Parameters
    ----------
    value : numpy.ndarray, pandas.DataFrame, or str
        Matrix to validate, or ``'identity'`` to indicate an identity
        matrix.
    name : str
        Name of the matrix, used in error messages.

    Raises
    ------
    TypeError
        If the input is not a supported type or contains non-numeric values.
    ValueError
        If a string other than ``'identity'`` is provided, if the matrix
        fails square-matrix validation, or if any off-diagonal entry
        is nonzero.

    Notes
    -----
    The ``'identity'`` sentinel is accepted without constructing a matrix.
    Unlike the symmetry check, the diagonal check uses exact equality
    and does not permit nonzero numerical tolerances.
    """
    if isinstance(value, str):
        if value != "identity":
            raise ValueError(
                f"{name} must be a square matrix or 'identity'."
            )
        return

    _validate_square_matrix(value, name)

    array = np.asarray(value)

    # All entries besides diagonal must be zero.
    if np.any(array != np.diag(np.diag(array))):
        raise ValueError(f"{name} must be diagonal.")


def _validate_A(A):
    """Validate an adjacency matrix representing an undirected network.

    Require a finite, numeric, symmetric square NumPy array or pandas
    DataFrame. Symmetry ensures that the connection weight from node
    ``i`` to node ``j`` matches the weight from ``j`` to ``i``.

    Parameters
    ----------
    A : numpy.ndarray or pandas.DataFrame
        Adjacency matrix describing connections between network nodes.

    Raises
    ------
    TypeError
        If the adjacency matrix has an unsupported type or contains
        non-numeric values.
    ValueError
        If the adjacency matrix is not two-dimensional, square, finite,
        or symmetric, or if its DataFrame labels do not match.
    """
    _validate_square_matrix(A, "A")


def _validate_B(B):
    """Validate a diagonal control input matrix or identity sentinel.

    Require an explicit control input matrix to be finite, numeric,
    square, symmetric, and diagonal. Alternatively, accept ``'identity'``
    to represent an identity control input matrix.

    Parameters
    ----------
    B : numpy.ndarray, pandas.DataFrame, or str
        Control input matrix or the string ``'identity'``.

    Raises
    ------
    TypeError
        If the input has an unsupported type or contains non-numeric values.
    ValueError
        If the input is neither ``'identity'`` nor a valid diagonal
        matrix.
    """
    _validate_square_matrix_or_identity(B, "B")


def _validate_S(S, energy_type):
    """Validate the state-penalty matrix for the selected energy type.

    The accepted value of ``S`` depends on the energy formulation.
    Minimal energy requires ``S=None``. Optimal energy requires either
    a finite, numeric, symmetric diagonal matrix or the string
    ``'identity'``.

    Parameters
    ----------
    S : numpy.ndarray, pandas.DataFrame, str, or None
        State-penalty matrix, the string ``'identity'``, or ``None``,
        depending on the selected energy type.
    energy_type : {'minimal', 'optimal'}
        Energy formulation determining the requirements for ``S``.

    Raises
    ------
    TypeError
        If an explicit matrix has an unsupported type or contains
        non-numeric values.
    ValueError
        If ``S`` is not ``None`` for minimal energy, is ``None`` for
        optimal energy, or fails matrix validation.

    Notes
    -----
    This function assumes that ``energy_type`` has already been validated.
    Values other than ``'minimal'`` and ``'optimal'`` are not handled
    explicitly.
    """
    if energy_type == "minimal":
        if S is not None:
            raise ValueError("S must be None when energy_type='minimal'.")
    elif energy_type == "optimal":
        if S is None:
            raise ValueError("S must have a value when energy_type='optimal'.")
        _validate_square_matrix_or_identity(S, "S")


def _validate_same_shape_matrices(matrices, names):
    """Validate that a collection of named matrices has matching shapes.

    Compare the shapes of all supplied matrices and raise an error if
    any matrix differs in shape from the others. Matrix names are used
    to identify the individual shapes in the error message.

    Parameters
    ----------
    matrices : iterable of array-like
        Matrices to compare. Each object must provide a ``shape``
        attribute.
    names : iterable of str
        Names corresponding to the matrices, in the same order.

    Raises
    ------
    ValueError
        If the number of matrix names differs from the number of
        matrices, or if the matrices have different shapes.

    Notes
    -----
    This function compares shapes only. It does not validate matrix
    dimensionality, numeric values, symmetry, or diagonality.

    An empty collection or a collection containing only one matrix
    passes validation.
    """
    matrices = tuple(matrices)
    names = tuple(names)

    if len(matrices) != len(names):
        raise ValueError("matrices and names must have the same length.")

    shapes = {matrix.shape for matrix in matrices}
    if len(shapes) > 1:
        formatted_shapes = ", ".join(
            f"{name}: {matrix.shape}"
            for name, matrix in zip(names, matrices)
        )
        raise ValueError(
            f"{', '.join(names)} must have the same shape; "
            f"got {formatted_shapes}"
        )