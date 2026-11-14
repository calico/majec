"""MAJEC - Momentum-Accelerated Junction-Enhanced Counting."""

# Source of truth for the version of the code actually being imported.
# The env is an editable install (site-packages/majec.pth -> this tree), so the
# installed dist-info metadata goes stale after a bump; this literal does not.
# Keep in sync with pyproject.toml.
__version__ = "0.2.0"
