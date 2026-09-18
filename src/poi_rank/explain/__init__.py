"""Explainability (spec.md section 10): grouped TreeSHAP, deterministic
natural-language templates, and a genuine compatibility counterfactual. No LLM
anywhere in this package -- spec.md section 10/brief section 4 explicitly rule
LLM-based explanation generation out of scope; everything here is either a real
numeric computation (SHAP, re-scoring) or a template string with numeric fill-ins.

**Firewall**: `explain/` is the LAST stage before output -- it MAY import from
`poi_rank.data`, `poi_rank.features`, `poi_rank.candidates`, `poi_rank.models`, and
`poi_rank.scoring` (every upstream phase's output/machinery this phase consumes),
but must never reference the oracle-only export directory (`tests/
test_firewall_explain.py`, mirroring every other non-`datagen`/`eval` module's
oracle isolation). Nothing is downstream of `explain/`, so there is no
"never imports X" half of the firewall the way `scoring/`/`models/` have one against
each other -- only the oracle-isolation half applies here.
"""

from __future__ import annotations
