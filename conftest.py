"""Make the test suite import *this* repo's majec.

The development repo installs itself into the shared conda env with a .pth file
pointing at MAJEC_project/src. That entry is on sys.path for every interpreter
in the env, so a bare `pytest tests/` here imported the dev tree instead --
0.2.0 code exercised by the release repo's tests, silently, with everything
passing. Putting src/ at the front of sys.path takes the ambiguity out.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'src'))
