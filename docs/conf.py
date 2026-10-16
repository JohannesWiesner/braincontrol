"""Sphinx configuration for the braincontrol documentation."""

from importlib.metadata import version
from pathlib import Path
import shutil

project = "braincontrol"
author = "Johannes Wiesner"
release = version("braincontrol")

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_autodoc_typehints",
    "myst_nb",
]

html_theme = "sphinx_rtd_theme"
html_title = f"{project} {release} documentation"

# The canonical tutorial stays in examples/ for Binder and local use.
# Copy it into the Sphinx source tree so MyST-NB can render it.
docs_dir = Path(__file__).resolve().parent
tutorial_source = docs_dir.parent / "examples" / "braincontrol_tutorial.ipynb"
tutorial_destination = docs_dir / "_generated" / "braincontrol_tutorial.ipynb"
tutorial_destination.parent.mkdir(parents=True, exist_ok=True)
shutil.copyfile(tutorial_source, tutorial_destination)

# Keep Read the Docs builds fast and deterministic. Notebook outputs
# are displayed only if they have been saved in the source notebook.
nb_execution_mode = "off"
exclude_patterns = ["_build"]

html_baseurl = __import__("os").environ.get("READTHEDOCS_CANONICAL_URL", "")
