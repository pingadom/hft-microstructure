"""Turn notebooks/*.py (the source you edit) into executed notebooks/*.ipynb (what GitHub shows).

The .py files use the "percent" format: `# %%` starts a code cell, `# %% [markdown]` a text
cell. They're plain Python, so they diff nicely in git and open in any editor. VS Code and
PyCharm can also run them cell by cell directly.

Usage:
    .venv\\Scripts\\python scripts\\build_notebooks.py            # all chapters
    .venv\\Scripts\\python scripts\\build_notebooks.py 05 06      # just these
"""

import sys
import time
from pathlib import Path

import jupytext
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

NB_DIR = Path(__file__).resolve().parents[1] / "notebooks"


def build(src: Path) -> None:
    t0 = time.time()
    nb = jupytext.read(src)
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    ExecutePreprocessor(timeout=1800, kernel_name="python3").preprocess(nb, {"metadata": {"path": str(NB_DIR)}})
    nbformat.write(nb, src.with_suffix(".ipynb"))
    print(f"built {src.with_suffix('.ipynb').name} in {time.time() - t0:.0f}s", flush=True)


def main() -> None:
    wanted = sys.argv[1:]
    for src in sorted(NB_DIR.glob("[0-9][0-9]_*.py")):
        if not wanted or src.name[:2] in wanted:
            build(src)


if __name__ == "__main__":
    main()
