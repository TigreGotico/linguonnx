"""Guard against the exact failure mode this suite already shipped once.

``test/test_translate_integration.py`` sat behind
``pytest.importorskip("optimum.onnxruntime")`` while CI's ``test`` extra
installed nothing but ``pytest`` and ``onnx``. Every run reported a plain
"9 skipped" and stayed green; nobody looked twice, because nothing about a
low, stable skip count says "an entire module never ran." linguonnx#44's
regression shipped behind exactly that guard.

This does not try to forbid skips outright - some are genuinely deliberate
(a network-only test suite excluded from the offline run; a registry gap
that is tracked elsewhere, not a missing dependency). It instead pins each
*known* skip reason to the exact count it is expected at, so that:

* a new, undocumented skip reason fails the run instead of blending into
  the total, and
* a documented reason silently growing - the module-skip pattern that hid
  linguonnx#44 - fails the run too, instead of "N skipped" quietly becoming
  "N + 30 skipped" with nobody noticing.

The check only applies to a full-suite run (see ``_looks_like_a_full_run``
below): a developer running one file or one ``-k`` selection during normal
work should not have to reason about global skip accounting.
"""

from __future__ import annotations

# reason substring -> the exact number of skips it may account for.
ALLOWED_SKIPS: dict[str, int] = {
    # test_translate_integration.py: real weights, real HF downloads, and a
    # torch-backed `optimum`+`transformers` reference. Correctly excluded from
    # the offline "not network" run by `pytest.mark.network` - but the
    # `importorskip` guarding the reference imports runs at *collection* time,
    # before pytest ever gets to look at the marker, so a missing `optimum`
    # always shows up as one module-wide "skipped", never as 14 individual
    # "deselected" tests. `test/test_translate_pipeline_isolation.py` and
    # `test/test_translate_native_codes.py` carry offline equivalents of the
    # behaviours this module guards; when `optimum` *is* installed (`-m
    # network`, real internet access), this line collects and runs instead.
    "reference outputs need optimum": 1,
    # test_translate_pivot_distance.py: the registry does not yet have a
    # dedicated pt->es capability to route through - a feature gap, filed
    # separately, not a missing test dependency.
    "no dedicated pt->es capability in the registry yet": 1,
}

#: Below this many collected items, this is a developer running a subset
#: (one file, one `-k`), not the full suite CI runs - skip accounting does
#: not apply.
_FULL_RUN_FLOOR = 500


def _looks_like_a_full_run(session) -> bool:
    return len(session.items) >= _FULL_RUN_FLOOR


def _skip_reason(report) -> str:
    longrepr = report.longrepr
    if isinstance(longrepr, tuple) and len(longrepr) == 3:
        return str(longrepr[2])
    return str(longrepr)


def pytest_sessionfinish(session, exitstatus):
    if not _looks_like_a_full_run(session):
        return

    terminalreporter = session.config.pluginmanager.get_plugin("terminalreporter")
    if terminalreporter is None:
        return

    skipped = terminalreporter.stats.get("skipped", [])
    seen: dict[str, int] = {}
    unmatched: list[str] = []
    for report in skipped:
        reason = _skip_reason(report)
        for allowed_reason in ALLOWED_SKIPS:
            if allowed_reason in reason:
                seen[allowed_reason] = seen.get(allowed_reason, 0) + 1
                break
        else:
            unmatched.append(f"{report.nodeid}: {reason}")

    problems = list(unmatched)
    for reason, expected in ALLOWED_SKIPS.items():
        actual = seen.get(reason, 0)
        if actual != expected:
            problems.append(
                f"expected exactly {expected} skip(s) for {reason!r}, "
                f"got {actual}")

    if problems:
        terminalreporter.section("Unexpected skip accounting")
        for problem in problems:
            terminalreporter.write_line(problem)
        terminalreporter.write_line(
            "A skip that is not on the ALLOWED_SKIPS allow-list in "
            "test/conftest.py, or a documented one that grew, is exactly how "
            "linguonnx#44 shipped behind a green suite. Add the dependency "
            "the skip is missing (preferred - see the `test` extra in "
            "pyproject.toml) or update ALLOWED_SKIPS with a reason for why "
            "this one has to stay skipped and cannot grow further.")
        session.testsfailed += 1
        session.exitstatus = 1
