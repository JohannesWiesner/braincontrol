`<img src="assets/logo.svg" alt="braincontrol logo" width="400">`{=html}

# braincontrol

`braincontrol` provides Network Control Theory (NCT) tools for
neuroimaging data. Its main estimator, `Transitioner`, computes
node-level control energy for transitions between brain states using
`nctpy`, with support for NumPy, pandas, and Niimg-like state inputs.

## Installation

``` bash
pip install braincontrol
```

`braincontrol` requires Python 3.9 or newer.

## Quick start

`Transitioner` follows a scikit-learn-style `fit` / `transform`
workflow. Estimator parameters such as the time horizon and transition
strategy are configured when the estimator is created. Empirical
matrices and states are supplied to `fit`, `transform`, or
`fit_transform`.

``` python
import numpy as np

from braincontrol.transitions import Transitioner

# Example network with three nodes.
A = np.array([
    [0.0, 0.4, 0.2],
    [0.4, 0.0, 0.3],
    [0.2, 0.3, 0.0],
])

# Rows are states, columns are nodes.
X = np.array([
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
])

transitioner = Transitioner(
    T=1,
    transitions="directed",
)

energy = transitioner.fit_transform(
    A=A,
    X=X,
)

print(energy)
```

For unlabeled NumPy input, the returned energy has shape
`(n_transitions, n_nodes)`. If node or state labels are available,
`Transitioner` preserves them in a pandas `DataFrame`.

## State input

State input can be supplied in two ways:

1.  A single state set `X`.
2.  Separate initial and final state sets `X0` and `Xf`.

These representations cannot be mixed.

For two-dimensional tabular input, rows represent states and columns
represent nodes. A one-dimensional input represents a single state.

### Single state set

``` python
transitioner = Transitioner(
    T=1,
    transitions="directed",
)

energy = transitioner.fit_transform(
    A=A,
    X=X,
)
```

The available transition strategies depend on the number of states in
`X`:

  -------------------------------------------------------------------------------
  Strategy                 One state         Multiple states   Meaning
  ------------------------ ----------------- ----------------- ------------------
  `"directed"`             No                Yes               All directed
                                                               transitions
                                                               between distinct
                                                               states

  `"undirected"`           No                Yes               One transition for
                                                               each pair of
                                                               distinct states

  `"directed_with_self"`   No                Yes               All directed
                                                               transitions,
                                                               including
                                                               self-transitions

  `"self"`                 Yes               Yes               One
                                                               self-transition
                                                               for each state
  -------------------------------------------------------------------------------

For a single state, `"self"` is the only valid strategy.

``` python
single_state = np.array([1.0, 0.0, 0.0])

transitioner = Transitioner(
    T=1,
    transitions="self",
)

energy = transitioner.fit_transform(
    A=A,
    X=single_state,
)
```

### Separate initial and final states

Use `X0` and `Xf` when initial and final states are represented by
separate state sets.

``` python
X0 = np.array([
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
])

Xf = np.array([
    [0.0, 0.0, 1.0],
    [1.0, 0.0, 0.0],
])

transitioner = Transitioner(
    T=1,
    transitions="paired",
)

energy = transitioner.fit_transform(
    A=A,
    X0=X0,
    Xf=Xf,
)
```

Two strategies are available:

-   `"all_to_all"` computes every transition from a state in `X0` to a
    state in `Xf`.
-   `"paired"` computes transitions between corresponding states in `X0`
    and `Xf`. The two inputs must contain the same number of states.

## Fit and transform

`fit` validates the estimator configuration and empirical inputs and
learns the node schema:

``` python
transitioner.fit(
    A=A,
    X=X,
)
```

The fitted schema is stored in:

``` python
transitioner.n_nodes_
transitioner.node_labels_
```

`transform` validates new inputs against that fitted schema and computes
transition energy:

``` python
energy = transitioner.transform(
    A=A,
    X=X,
)
```

When fitting and transforming the same data, use:

``` python
energy = transitioner.fit_transform(
    A=A,
    X=X,
)
```

## Matrix inputs

`A` is the adjacency matrix and must have shape `(n_nodes, n_nodes)`.

The control input matrix `B` defaults to `"identity"`:

``` python
energy = transitioner.fit_transform(
    A=A,
    B="identity",
    X=X,
)
```

For optimal control energy, the state-trajectory constraint matrix `S`
also defaults to `"identity"`.

All empirical matrix and state inputs must describe the same number of
nodes.

## Adjacency matrix normalization

By default, `Transitioner` normalizes `A` during `transform` using
`nctpy.utils.matrix_normalization` for the selected system.

``` python
transitioner = Transitioner(
    T=1,
    system="continuous",
    c=2,
)
```

To use the adjacency matrix as supplied:

``` python
transitioner = Transitioner(
    T=1,
    normalize_A=False,
)
```

## Energy type

Optimal control energy is the default:

``` python
transitioner = Transitioner(
    T=1,
    energy_type="optimal",
    rho=1.0,
)
```

For optimal energy, `rho`, `S`, and `xr` participate in the control
problem. `S` defaults to `"identity"` and `xr` defaults to `"xf"`.

Minimal control energy is selected explicitly:

``` python
transitioner = Transitioner(
    T=1,
    energy_type="minimal",
    rho=None,
)

energy = transitioner.fit_transform(
    A=A,
    S=None,
    X=X,
    xr=None,
)
```

For minimal energy, `rho`, `S`, and `xr` must be compatible with the
minimal-energy configuration.

## Reference state

For optimal energy, `xr` can be one of the supported symbolic reference
states:

``` python
energy = transitioner.fit_transform(
    A=A,
    X=X,
    xr="xf",
)
```

Supported string values are:

-   `"zero"`
-   `"x0"`
-   `"xf"`
-   `"midpoint"`

A custom reference state can also be supplied as a supported empirical
state input. The reference state must describe exactly one state.

## Labels and schema

`braincontrol` preserves labels when they can be inferred from pandas
inputs.

For a `DataFrame`:

-   rows represent states;
-   the index provides state labels;
-   columns provide node labels.

``` python
import pandas as pd

states = pd.DataFrame(
    [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ],
    index=["rest", "task", "recovery"],
    columns=["node_A", "node_B", "node_C"],
)

adjacency = pd.DataFrame(
    A,
    index=states.columns,
    columns=states.columns,
)

transitioner = Transitioner(
    T=1,
    transitions="directed",
)

energy = transitioner.fit_transform(
    A=adjacency,
    X=states,
)
```

Node labels form part of the fitted schema. Transform-time node labels
must therefore be compatible with the labels learned during `fit`.

State labels are transform-time metadata rather than fitted schema. They
are used to label the transitions produced by the current transform
call.

A pandas `Series` represents a single state: its index provides node
labels, and its `name`, when present, provides the state label.

Pandas `MultiIndex` objects are supported for hierarchical node and
state metadata.

### Separate `X0` and `Xf` labels

When `X0` and `Xf` are supplied separately, they must either both
provide state labels or both be unlabeled. If hierarchical state labels
are represented by a `MultiIndex`, `X0` and `Xf` must use compatible
MultiIndex structures.

## Neuroimaging input

State inputs may be Niimg-like objects when a compatible Nilearn masker
is supplied.

``` python
from nilearn.maskers import NiftiLabelsMasker

masker = NiftiLabelsMasker(
    labels_img=atlas,
)

transitioner = Transitioner(
    T=1,
    transitions="directed",
    masker=masker,
)

energy = transitioner.fit_transform(
    A=A,
    X=states_img,
)
```

For each Niimg-like input, `Transitioner` clones the configured masker,
fits it on that input, and transforms the image data into a node-level
array. The user-provided masker instance itself is not fitted in place.

The resulting node representation must be compatible with the node
schema established from the other empirical inputs.

## Storing trajectories

State and control trajectories can optionally be retained during
`transform`:

``` python
transitioner = Transitioner(
    T=1,
    transitions="directed",
    store_state_trajectories=True,
    store_control_trajectories=True,
)

energy = transitioner.fit_transform(
    A=A,
    X=X,
)
```

Retrieve them with:

``` python
state_trajectories = transitioner.get_state_trajectories()
control_trajectories = transitioner.get_control_trajectories()
```

The getters return `xarray.DataArray` objects with dimensions:

``` text
("time", "node", "transition")
```

Available node and transition labels are included as coordinates. If the
corresponding storage option is disabled, the getter returns `None`.

## Numerical errors

Numerical errors reported by `nctpy` for the most recent transform are
available with:

``` python
errors = transitioner.get_errors()
```

## Main `Transitioner` parameters

``` python
Transitioner(
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
)
```

The empirical inputs `A`, `B`, `S`, `X`, `X0`, `Xf`, and `xr` are
supplied to `fit`, `transform`, or `fit_transform`, rather than to the
constructor.

## Development

Install the development dependencies and run the test suite with:

``` bash
python -m pytest -q
```

## License

`braincontrol` is distributed under the MIT License.
