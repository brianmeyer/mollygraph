# Overnight Review Report

**Date:** 2026-02-26  
**Reviewer:** Molly (Sonnet 4.6)  
**Commits reviewed:** `978b7e3`, `b09a832`  
**Test run:** 46/46 passed (full suite)

---

## Summary

Two clean, well-scoped changes. No regressions. One low-risk edge case flagged in the normalization logic. No patches required.

---

## Focus Area Results

### 1. Atomic Persistence — `metrics/model_health.py` ✅ PASS

**What changed:** Replaced the old `.with_suffix(".tmp")` write-then-rename approach with `tempfile.mkstemp` + `os.fdopen` + `os.replace`. Tmp file is cleaned up on exception.

**Review findings:**
- `mkstemp(dir=MODEL_HEALTH_STATE_PATH.parent)` guarantees same-filesystem placement, so `os.replace` is a true atomic rename. Correct.
- `ensure_ascii=True` added as a bonus — avoids encoding surprises on reload. Good hygiene.
- Exception handler: temp file is unlinked on failure before re-raising, then the outer `except` swallows with a warning (not a crash). Appropriate for a metrics side-path.
- **Thread safety note (pre-existing, not introduced here):** `record_extraction` does `deque.append` + counter increment without a lock. Under CPython/GIL this is safe in practice, but not technically thread-safe. Not introduced by these commits; flagged as a low-priority hardening item.

**Tests:** `test_model_health_persists_windows_and_supports_legacy_state` covers round-trip persistence, legacy state compatibility, and no-`.tmp`-residue assertion. ✅

---

### 2. Kellogg Normalization — `evolution/contact_training.py` ✅ PASS (one edge case noted)

**What changed:** Introduced `RelationshipNoteNormalization` dataclass + `_normalize_relationship_note()`. Kellogg aliases → canonical `"Kellogg School of Management"`. Campus and cohort extracted. Classmate assertion only emitted on `high_confidence=True`. Canonical school entity injected into NER spans.

**Review findings:**
- Kellogg alias matching is substring (`in lowered`), not word-boundary. `"northwestern university - kellogg"` alias handles hyphen-spaced variants. Aliases are conservative enough that false-positives are unlikely.
- Campus matching: `_CAMPUS_ALIASES` uses first-match logic (dict order). Fine — the aliases are non-overlapping.
- Cohort regex `\bclass\s+of\s+(20\d{2})\b` matched against `lowered`; captures `Class of YYYY` correctly in the formatted output. ✅
- `_build_entity_spans` now adds `relationship_note.school` as an `Organization` span only when present — correctly gated. ✅
- **Edge case — `has_classmate_signal` without Kellogg:** If a note contains `alumni`/`classmate`/`cohort`/`emba` but no Kellogg keyword (e.g., `"HBS alumni connection"`), the code falls to `if has_classmate_signal: return RelationshipNoteNormalization(text=raw, high_confidence=True)`. This emits the raw note verbatim as a high-confidence assertion. Not a hallucination risk (raw text preserved, not fabricated), but it's a precision issue — non-Kellogg alumni notes get promoted to relationship sentences with the same confidence level as properly normalized Kellogg notes. Low severity for current data volume.

**Tests:** Two new tests in `test_contact_training_normalization.py`:
- Positive: canonical school + campus + cohort in reformatted text, plus entity span. ✅
- Negative: low-signal LinkedIn note suppressed. ✅
- Missing coverage: `alumni` alone (no Kellogg) → high-confidence raw emission (see edge case above). Low priority given current data profile.

---

### 3. Regression Check — Full Suite ✅ PASS

```
46 passed, 9 warnings in 23.33s
```

All pre-existing tests pass. Warnings are all PyTorch 3.14 deprecation noise (`torch.jit.script`), pre-existing and unrelated to these changes.

---

## Fixes Applied

None. No patches required.

---

## Remaining Risks

| Risk | Severity | Area | Action |
|---|---|---|---|
| `has_classmate_signal` without Kellogg emits raw note as high-confidence | Low | `contact_training.py` | Add test; consider requiring `has_kellogg OR explicit school name` for high-confidence path |
| No lock on `record_extraction` deque+counter (pre-existing) | Low | `model_health.py` | Add `threading.Lock` if/when concurrent extraction load increases |
| PyTorch 3.14 `torch.jit.script` deprecation (pre-existing) | Low | `.venv` dependency | Monitor; no immediate impact |

---

## Verdict

Both commits are solid. The atomic write hardening is straightforward and correct. The normalization work is well-structured with appropriate confidence gating. The one noted edge case is worth a follow-up test but poses no immediate data integrity risk.

**Overall: PASS — safe to leave in prod.**
