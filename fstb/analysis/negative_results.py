# Negative results analysis for Phase 3
# ---------------------------------------------------
# This module inspects the aggregated result dictionary produced by the
# Phase‑3 experiment runner and automatically generates a concise markdown (or
# JSON) section describing any tasks where the FSTB model fails to achieve a
# statistically significant improvement over the baseline.
#
# The function is deliberately lightweight – it expects the result dict to
# contain a ``stat_tests`` entry (produced by ``StatisticalSignificanceAnalyzer``)
# with a ``p_value`` field for each evaluated metric.  If a metric's p‑value is
# >= 0.05 we treat it as a non‑significant result and add it to the negative
# results report.
#
# The output format matches what :class:`PaperGenerator` expects for its
# "limitations" section (a markdown string).  Users can customise the string
# handling in the generator if they need LaTeX instead.

from typing import Dict, List, Any


def _format_metric(name: str, p_val: float) -> str:
    """Return a markdown bullet for a non‑significant metric.

    ``name`` – human readable metric name (e.g., "Memory F1").
    ``p_val`` – p‑value from statistical test.
    """
    return f"- **{name}**: p‑value = {p_val:.3f} (no significant gain)"


def generate_negative_results_section(aggregated_results: Dict[str, Any]) -> str:
    """Generate a markdown section listing all non‑significant outcomes.

    Parameters
    ----------
    aggregated_results : dict
        The full result dictionary returned by the Phase‑3 experiment runner.
        Expected keys:
        - ``stat_tests`` – mapping metric names to a dict containing at least
          ``p_value``.
        - ``negative_results`` – will be populated with the generated string.

    Returns
    -------
    str
        Markdown formatted list of metrics where FSTB did not show a
        statistically significant improvement.
    """
    stat_tests = aggregated_results.get("stat_tests", {})
    if not stat_tests:
        return "*No statistical test information available.*"

    non_sig: List[str] = []
    for metric_name, test_info in stat_tests.items():
        p_val = test_info.get("p_value")
        if p_val is None:
            continue
        if p_val >= 0.05:
            non_sig.append(_format_metric(metric_name.replace("_", " ").title(), p_val))

    if not non_sig:
        return "*All evaluated metrics achieved statistical significance (p < 0.05).*
"
    header = "## Negative Results and Limitations\n\nThe following metrics did **not** reach statistical significance (p ≥ 0.05):\n"
    return header + "\n".join(non_sig) + "\n"
