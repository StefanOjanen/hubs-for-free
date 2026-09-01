# Delegates to scale_round_v2.py (vectorized implementation of the same
# registered protocol) so notebooks pinned to this filename run v2.
# Session-1 history and the deviation rationale: RUNLOG.md. The original
# v1 implementation is preserved at commit 9dd433a.
import runpy

print("scale_round.py delegating to scale_round_v2.py (see RUNLOG.md)", flush=True)
runpy.run_path("alignment_study/scale_round_v2.py", run_name="__main__")
