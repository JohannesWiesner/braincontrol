# Installation

Install the released package with pip:

```bash
python -m pip install braincontrol
```

To try the current development branch:

```bash
python -m pip install "git+https://github.com/JohannesWiesner/braincontrol.git@dev"
```

## Quick start

```python
import numpy as np
from braincontrol.transitions import Transitioner

A = np.array([
    [0.0, 0.4, 0.2],
    [0.4, 0.0, 0.3],
    [0.2, 0.3, 0.0],
])
X = np.eye(3)

transitioner = Transitioner(T=1.0, transitions="directed")
energy = transitioner.fit_transform(A=A, X=X)
print(energy)
```

For a longer introduction, see the [tutorial](_generated/braincontrol_tutorial).
