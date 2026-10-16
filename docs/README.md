# Building the documentation

From the repository root:

```bash
python -m pip install -r docs/requirements.txt
python -m sphinx -b html docs docs/_build/html
```

Open `docs/_build/html/index.html` in a browser.

The Sphinx configuration copies the canonical
`examples/braincontrol_tutorial.ipynb` into `docs/_generated/` at build
time. Do not edit the generated copy. Notebook execution is disabled for
documentation builds; save notebook outputs in the source notebook if you
want them displayed on Read the Docs.

After merging this branch, import the GitHub repository into Read the Docs
Community at https://app.readthedocs.org/ and select the desired default
documentation branch. Read the Docs will discover `.readthedocs.yaml`.
