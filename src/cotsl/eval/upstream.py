"""Imports from the pinned upstream submodules (external/CoTControl, external/reasonIF).

Graders are used unmodified so binary compliance is byte-identical to the benchmark's own.
"""

from __future__ import annotations

import sys

from cotsl.paths import COTCONTROL_QA, REASONIF_SRC

if not COTCONTROL_QA.exists():
    raise ImportError("external/CoTControl missing: run `git submodule update --init --recursive`")

if str(COTCONTROL_QA) not in sys.path:
    sys.path.insert(0, str(COTCONTROL_QA))

# Re-exported for the eval modules.
import grading as cc_grading  # noqa: E402, F401
import prompts as cc_prompts  # noqa: E402, F401

# grade_compliance_csv calls load_dotenv() at import; harmless (we never override env).
import grade_compliance_csv as cc_judging  # noqa: E402, F401


def reasonif_checkers():
    """ReasonIF instruction checkers, with METR's fast_langdetect compatibility shim."""
    import fast_langdetect as fld

    if not getattr(fld.detect, "_cotsl_patched", False):
        orig = fld.detect

        def detect(text, **kwargs):
            kwargs.pop("low_memory", None)
            res = orig(text, **kwargs)
            if isinstance(res, list):
                return res[0] if res else {"lang": "unknown", "score": 0.0}
            return res

        detect._cotsl_patched = True
        fld.detect = detect
    if str(REASONIF_SRC) not in sys.path:
        sys.path.insert(0, str(REASONIF_SRC))
    from instructions import instruction_checker

    return instruction_checker
