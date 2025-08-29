# tests/test_workflow.py
import ast
import numpy as np
import pytest

# Import your public API exactly as before (no core edits required)
from NeuroCluster import run_cluster_workflow, plot_results


def _as_value(obj):
    """
    Normalize a 'cluster' or 'null' entry into a numeric stat.
    Accepts dicts, tuples/lists/ndarrays, stringified structures, or simple objects.
    """
    # dict-like
    if isinstance(obj, dict):
        for k in ("cluster_stat", "stat", "value"):
            if k in obj:
                return obj[k]
        raise AssertionError(f"Dict missing a recognized stat key: {obj}")

    # array/tuple-like: assume first entry holds the stat
    if isinstance(obj, (list, tuple, np.ndarray)) and len(obj) > 0:
        return obj[0]

    # stringified dict/tuple/list
    if isinstance(obj, str):
        try:
            parsed = ast.literal_eval(obj)
            return _as_value(parsed)
        except Exception as e:
            raise AssertionError(f"Unparseable stat string: {obj!r}") from e

    # object attributes
    if hasattr(obj, "cluster_stat"):
        return getattr(obj, "cluster_stat")
    if hasattr(obj, "stat"):
        return getattr(obj, "stat")

    # If we get here, we don't know how to read this object
    raise AssertionError(f"Unsupported stat type: {type(obj)} -> {obj!r}")


@pytest.fixture(scope="module")
def workflow_results():
    """
    Run the small/fast workflow your tests already rely on.
    Keep the invocation identical to your current tests—only consumption changes.
    """
    # If your existing tests pass specific args, keep them here.
    # Otherwise, call with defaults as your test previously did.
    return run_cluster_workflow()


def test_cluster_significance_test(workflow_results):
    """
    Previously:
        for cluster, null_stats in zip(max_cluster_data, null_distribution):
            assert np.sign(cluster['cluster_stat']) == np.sign(null_stats[0])
    Now we normalize both sides to be robust to return shape changes.
    """
    results = workflow_results
    assert "max_cluster_data" in results and "null_distribution" in results

    max_cluster_data = results["max_cluster_data"]
    null_distribution = results["null_distribution"]

    # Basic sanity on lengths
    assert len(max_cluster_data) == len(null_distribution), (
        f"Length mismatch: {len(max_cluster_data)} vs {len(null_distribution)}"
    )

    for cluster, null_stats in zip(max_cluster_data, null_distribution):
        cluster_stat = _as_value(cluster)
        null_stat0 = _as_value(null_stats)

        # Both should be numeric
        assert isinstance(cluster_stat, (int, float, np.floating)), (
            f"Non-numeric cluster stat: {type(cluster_stat)} -> {cluster_stat!r}"
        )
        assert isinstance(null_stat0, (int, float, np.floating)), (
            f"Non-numeric null stat: {type(null_stat0)} -> {null_stat0!r}"
        )

        # Preserve the original intent: compare signs
        assert np.sign(cluster_stat) == np.sign(null_stat0), (
            f"Sign mismatch: cluster={cluster_stat}, null0={null_stat0}"
        )


def test_plot_results(workflow_results, tmp_path):
    """
    Keep the same call to plot_results as your original test.
    Only the post-conditions get more flexible about the structure returned.
    """
    results = workflow_results
    fig_path = tmp_path / "out.png"

    # Call exactly as before (kwargs preserved in case the function signature expects them)
    plot_results(
        clusters=results["max_cluster_data"],
        null=results["null_distribution"],
        save_path=fig_path,
    )

    # File should have been created
    assert fig_path.exists(), "The plot file was not created."

    # Be robust about cluster structure in assertions
    stats = [_as_value(c) for c in results["max_cluster_data"]]
    assert all(isinstance(s, (int, float, np.floating)) for s in stats), (
        f"Some cluster stats are not numeric: {stats}"
    )


