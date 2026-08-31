"""Keep the three declarations of the package version from drifting apart.

__init__.py froze at 0.1.2 while pyproject.toml went on to 0.1.3, 0.1.4 and
0.1.5. Nothing caught it because nothing reads __version__ yet -- it is a
decorative constant here, and only starts mislabeling anything once the version
is recorded in logs and run manifests.

Deriving __version__ from importlib.metadata is the usual fix and does not work
in this environment: the shared conda env carries a stale majec-0.1.0.dist-info,
so the metadata reports a version belonging to neither the source tree nor any
current release. Comparing the declarations to each other does not depend on
install state at all.
"""

import re
import tomllib
from pathlib import Path

import pytest

import majec

ROOT = Path(__file__).parent.parent


def test_version_matches_pyproject():
    with open(ROOT / 'pyproject.toml', 'rb') as fh:
        declared = tomllib.load(fh)['tool']['poetry']['version']
    assert majec.__version__ == declared, (
        f'majec.__version__ is {majec.__version__} but pyproject.toml declares '
        f'{declared} -- bump both or neither')


def test_version_matches_conda_recipe():
    recipe = ROOT / 'conda-recipe' / 'meta.yaml'
    if not recipe.exists():
        pytest.skip('conda recipe not in this checkout')
    # The recipe fetches the sdist for whatever version it names, so a stale
    # value here builds a package from the wrong release rather than failing.
    match = re.search(r'{%\s*set\s+version\s*=\s*"([^"]+)"\s*%}', recipe.read_text())
    assert match, 'no `{% set version = "..." %}` line in conda-recipe/meta.yaml'
    assert match.group(1) == majec.__version__, (
        f'conda recipe builds {match.group(1)} but the package is '
        f'{majec.__version__}')
