<img src="assets/logo_new_v2.svg" alt="braincontrol logo" width="400">

# braincontrol

`braincontrol` provides Network Control Theory (NCT) tools for
neuroimaging data. Its main estimator, `Transitioner`, computes
node-level control energy for transitions between brain states using the
[nctpy](https://github.com/LindenParkesLab/nctpy) package, with support for NumPy, pandas, and Niimg-like state inputs.

## Installation

``` bash
pip install braincontrol
```

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
    T=1.0,
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

## Documentation

Read the [documentation on Read the Docs](https://braincontrol.readthedocs.io/en/latest/) for installation, the tutorial, and the API reference. The site will become available after the project is imported and its first build succeeds.

## Tutorial

You can find a tutorial here:

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/JohannesWiesner/braincontrol/dev?urlpath=lab/tree/examples/braincontrol_tutorial.ipynb)

## License

`braincontrol` is distributed under the MIT License.
