# braincontrol

Network Control Theory for Neuroimaging Data.

## State transitions

`braincontrol.transitions` computes all requested transitions between rows of a
state matrix. By default, `Transitioner` normalizes the adjacency matrix with
nctpy for the selected time system.

```python
from braincontrol.transitions import Transitioner

transitioner = Transitioner(
    A=adjacency,
    T=1,
    B="identity",
    order="permutations",
    system="continuous",
)
energy = transitioner.fit_transform(states)  # states: (n_states, n_nodes)
errors = transitioner.get_errors()
trajectories, control_inputs = transitioner.get_transition_arrays()
```

The fitted normalized matrix is available as `transitioner.A_`; the original
`A` is not modified. The normalization constant `c` is exposed and defaults to
`1`:

```python
transitioner = Transitioner(
    A=adjacency,
    T=1,
    system="continuous",
    c=2,
)
energy = transitioner.fit_transform(X=states)
```

Normalization uses `nctpy.utils.matrix_normalization`. To provide an adjacency
matrix that is already normalized, disable normalization explicitly:

```python
transitioner = Transitioner(
    A=normalized_adjacency,
    T=1,
    system="continuous",
    normalize_A=False,
)
```

For one directed transition, provide the source and target states separately.
Both arguments are required, and they cannot be combined with `X`:

```python
energy = transitioner.fit_transform(x0=source_state, xf=target_state)
```

This always computes only `x0 -> xf`. To compute transitions among several
states, pass the complete `(n_states, n_nodes)` matrix as `X`; `order` then
determines which state pairs are evaluated.

`energy_type="optimal"` is the default and requires both `rho` and `S`.
Minimal control energy is selected explicitly, with both optimal-energy
parameters disabled:

```python
transitioner = Transitioner(
    A=adjacency,
    T=1,
    energy_type="minimal",
    rho=None,
    S=None,
)
```

Mixing `energy_type="minimal"` with a non-`None` `rho` or `S` raises a
`ValueError`. Conversely, optimal energy rejects missing `rho` or `S`.

The same transformer accepts a 4D NIfTI image when given a Nilearn masker.
Each 3D volume represents one state:

```python
from nilearn.maskers import NiftiLabelsMasker

transitioner = Transitioner(
    A=adjacency,
    T=1,
    masker=NiftiLabelsMasker(labels_img=atlas),
)
energy = transitioner.fit_transform(states_img)
```

Using a masker as a component keeps the numerical API available to users with
non-neuroimaging arrays while allowing images to be masked directly.

### Caching

`Transitioner` inherits Nilearn's `CacheMixin`. Set `memory` to a directory or
a `joblib.Memory` instance to cache the expensive state-transition computation:

```python
transitioner = Transitioner(
    A=adjacency,
    T=1,
    memory="braincontrol_cache",
    memory_level=1,
)
```

Repeated calls with identical adjacency, states, control parameters, and
transition settings reuse the cached trajectories, control inputs, and errors.
Changing any of those inputs creates a separate cache entry. Caching is disabled
by default with `memory=None`. For image inputs, masking has its own cache and
can be configured through the supplied Nilearn masker.

### Hierarchical node metadata

Transition results can retain any number of node attributes. An existing,
named pandas `MultiIndex` can be passed directly and is preserved as the
columns in the returned DataFrame:

```python
import pandas as pd
from braincontrol.transitions import get_state_to_state_df

node_index = pd.MultiIndex.from_arrays(
    [
        ["association", "association", "sensory"],
        ["default", "default", "visual"],
        ["medial", "lateral", "occipital"],
        ["A", "B", "C"],
    ],
    names=["cortex", "network", "region", "node"],
)

result = get_state_to_state_df(
    energy,
    order="permutations",
    node_attributes=node_index,
    state_attributes=state_attributes,
)
```

For a single attribute, pass any list-like object with one value per node:

```python
result = get_state_to_state_df(
    energy,
    order="permutations",
    node_attributes=["node_A", "node_B", "node_C"],
)
```

A list-like sequence of equal-length tuples is also accepted and converted to
a `MultiIndex`. `node_attributes` is the only node metadata parameter.

### Hierarchical state metadata

`state_attributes` follows the same convention. A state `MultiIndex` can describe
any number of higher-order assignments:

```python
state_attributes = pd.MultiIndex.from_arrays(
    [
        ["baseline", "active", "baseline"],
        ["rest", "task", "recovery"],
    ],
    names=["condition", "state"],
)

result = get_state_to_state_df(
    energy,
    order="permutations",
    node_attributes=node_index,
    state_attributes=state_attributes,
)
```

This produces transition fields named `source_condition`, `target_condition`,
`source_state`, and `target_state`. Additional levels produce corresponding
source and target fields. Flat list-like state attributes retain the
traditional `source_state` and `target_state` fields. `state_attributes` is
the only state metadata parameter.
