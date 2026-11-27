"""Shared pytest setup for test_polls_helpers.py and tests/."""
import matplotlib

# Chart tests call plt.show(); a non-interactive backend keeps them from opening
# (and blocking on) windows.
matplotlib.use("Agg")
