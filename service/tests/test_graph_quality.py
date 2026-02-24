"""Graph quality regression test — golden set evaluation.

Loads ``tests/golden_set.json`` and runs each test case through the live
extraction pipeline.  Checks that expected entities and relations are found
with sufficient precision/recall.

Pass criteria (configurable via environment variables):
  - Overall entity recall   >= QUALITY_ENTITY_RECALL_MIN   (default 0.80 / 80%)
  - Overall relation recall >= QUALITY_RELATION_RECALL_MIN (default 0.60 / 60%)

Skip conditions:
  - ``gliner2`` is not importable (model not installed; e.g. CI without torch)
  - MOLLYGRAPH_SKIP_QUALITY_TESTS=1 is set in the environment

Running standalone (no pytest):
  python tests/test_graph_quality.py
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Any

# ── Path setup ─────────────────────────────────────────────────────────────────
# Allow running from the repo root or from the tests/ directory.
_HERE = Path(__file__).parent
_SERVICE_ROOT = _HERE.parent
if str(_SERVICE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SERVICE_ROOT))

# ── Skip guard ─────────────────────────────────────────────────────────────────
_SKIP_ENV = os.environ.get("MOLLYGRAPH_SKIP_QUALITY_TESTS", "").strip().lower()
_skip_reason: str | None = None

if _SKIP_ENV in {"1", "true", "yes"}:
    _skip_reason = "MOLLYGRAPH_SKIP_QUALITY_TESTS is set"
else:
    try:
        import gliner2  # type: ignore[import]   # noqa: F401
    except ImportError:
        _skip_reason = "gliner2 is not importable (torch/model not installed)"

_GOLDEN_SET_PATH = _HERE / "golden_set.json"

# ── Thresholds ─────────────────────────────────────────────────────────────────
ENTITY_RECALL_MIN = float(os.environ.get("QUALITY_ENTITY_RECALL_MIN", "0.80"))
RELATION_RECALL_MIN = float(os.environ.get("QUALITY_RELATION_RECALL_MIN", "0.60"))


# ── Helpers ────────────────────────────────────────────────────────────────────

def _fuzzy_entity_match(expected_name: str, extracted_entities: list[dict[str, Any]]) -> bool:
    """Return True if *expected_name* fuzzy-matches any extracted entity.

    Strategy: case-insensitive substring in either direction.  Strips
    punctuation before comparing to handle minor tokenisation differences.
    """
    needle = expected_name.strip().lower()

    def _clean(s: str) -> str:
        return re.sub(r"[^\w\s]", "", s).strip().lower()

    needle_clean = _clean(needle)

    for ent in extracted_entities:
        hay = _clean(str(ent.get("text") or ""))
        if not hay:
            continue
        if needle_clean in hay or hay in needle_clean:
            return True
    return False


def _normalize_rel_type(raw: str) -> str:
    """Normalise a raw relation label to uppercase_underscore form."""
    return re.sub(r"[\s\-]+", "_", str(raw or "").strip()).upper()


def _relation_type_matches(expected_pattern: str, extracted_relations: list[dict[str, Any]]) -> bool:
    """Return True if any extracted relation's type matches *expected_pattern* (regex)."""
    pattern = re.compile(expected_pattern, re.IGNORECASE)
    for rel in extracted_relations:
        label = _normalize_rel_type(str(rel.get("label") or ""))
        if pattern.fullmatch(label):
            return True
    return False


def _source_target_match(
    expected_source: str,
    expected_target: str,
    extracted_relations: list[dict[str, Any]],
    expected_pattern: str,
) -> bool:
    """Check if any extracted relation matches source, target, AND type pattern.

    Source/target matching uses the same fuzzy substring logic as entity matching.
    """
    pattern = re.compile(expected_pattern, re.IGNORECASE)

    def _clean(s: str) -> str:
        return re.sub(r"[^\w\s]", "", s).strip().lower()

    src_clean = _clean(expected_source)
    tgt_clean = _clean(expected_target)

    for rel in extracted_relations:
        head = _clean(str(rel.get("head") or ""))
        tail = _clean(str(rel.get("tail") or ""))
        label = _normalize_rel_type(str(rel.get("label") or ""))

        head_match = src_clean in head or head in src_clean
        tail_match = tgt_clean in tail or tail in tgt_clean
        type_match = bool(pattern.fullmatch(label))

        if head_match and tail_match and type_match:
            return True
    return False


# ── Core evaluation ────────────────────────────────────────────────────────────

def load_golden_set() -> list[dict[str, Any]]:
    with _GOLDEN_SET_PATH.open(encoding="utf-8") as f:
        return json.load(f)


def run_quality_check() -> dict[str, Any]:
    """Run the full golden-set quality evaluation.

    Returns a results dict with per-case breakdown and aggregate metrics.
    Raises ``RuntimeError`` if the extractor cannot be loaded.
    """
    import config

    # Force non-test mode for real extraction.
    _was_test_mode = getattr(config, "TEST_MODE", False)
    config.TEST_MODE = False  # type: ignore[attr-defined]

    try:
        from memory import extractor as memory_extractor

        golden = load_golden_set()
        case_results: list[dict[str, Any]] = []

        total_entities_expected = 0
        total_entities_found = 0
        total_relations_expected = 0
        total_relations_found = 0

        for tc in golden:
            tc_id = tc.get("id", "?")
            text = str(tc.get("input", ""))
            expected_entities = tc.get("expected_entities", [])
            expected_relations = tc.get("expected_relations", [])

            # Run extraction
            result = memory_extractor.extract(text)
            extracted_entities = result.get("entities", [])
            extracted_relations = result.get("relations", [])
            latency_ms = result.get("latency_ms", 0)

            # Check entities
            entity_hits: list[dict[str, Any]] = []
            entity_misses: list[str] = []
            for exp in expected_entities:
                found = _fuzzy_entity_match(exp["name"], extracted_entities)
                if found:
                    entity_hits.append(exp)
                else:
                    entity_misses.append(exp["name"])

            n_exp_ent = len(expected_entities)
            n_hit_ent = len(entity_hits)
            entity_recall = n_hit_ent / n_exp_ent if n_exp_ent > 0 else 1.0
            entity_precision = (
                n_hit_ent / len(extracted_entities) if extracted_entities else (1.0 if n_exp_ent == 0 else 0.0)
            )

            # Check relations
            relation_hits: list[dict[str, Any]] = []
            relation_misses: list[dict[str, Any]] = []
            for exp_rel in expected_relations:
                found = _source_target_match(
                    exp_rel["source"],
                    exp_rel["target"],
                    extracted_relations,
                    exp_rel["type_pattern"],
                )
                if found:
                    relation_hits.append(exp_rel)
                else:
                    relation_misses.append(exp_rel)

            n_exp_rel = len(expected_relations)
            n_hit_rel = len(relation_hits)
            relation_recall = n_hit_rel / n_exp_rel if n_exp_rel > 0 else 1.0

            total_entities_expected += n_exp_ent
            total_entities_found += n_hit_ent
            total_relations_expected += n_exp_rel
            total_relations_found += n_hit_rel

            case_results.append(
                {
                    "id": tc_id,
                    "description": tc.get("description", ""),
                    "input": text,
                    "entity_recall": round(entity_recall, 3),
                    "entity_precision": round(entity_precision, 3),
                    "relation_recall": round(relation_recall, 3),
                    "entities_expected": n_exp_ent,
                    "entities_found": n_hit_ent,
                    "entity_misses": entity_misses,
                    "relations_expected": n_exp_rel,
                    "relations_found": n_hit_rel,
                    "relation_misses": [
                        f"{r['source']} --[{r['type_pattern']}]--> {r['target']}"
                        for r in relation_misses
                    ],
                    "extracted_entity_count": len(extracted_entities),
                    "extracted_relation_count": len(extracted_relations),
                    "latency_ms": latency_ms,
                }
            )

        overall_entity_recall = (
            total_entities_found / total_entities_expected if total_entities_expected > 0 else 1.0
        )
        overall_relation_recall = (
            total_relations_found / total_relations_expected if total_relations_expected > 0 else 1.0
        )

        passed = (
            overall_entity_recall >= ENTITY_RECALL_MIN
            and overall_relation_recall >= RELATION_RECALL_MIN
        )

        return {
            "passed": passed,
            "overall_entity_recall": round(overall_entity_recall, 4),
            "overall_relation_recall": round(overall_relation_recall, 4),
            "entity_recall_threshold": ENTITY_RECALL_MIN,
            "relation_recall_threshold": RELATION_RECALL_MIN,
            "total_test_cases": len(golden),
            "total_entities_expected": total_entities_expected,
            "total_entities_found": total_entities_found,
            "total_relations_expected": total_relations_expected,
            "total_relations_found": total_relations_found,
            "cases": case_results,
            "failures": [
                c for c in case_results if c["entity_recall"] < ENTITY_RECALL_MIN
            ],
        }
    finally:
        config.TEST_MODE = _was_test_mode  # type: ignore[attr-defined]


# ── pytest integration ─────────────────────────────────────────────────────────

try:
    import pytest

    _skip_mark = pytest.mark.skipif(
        _skip_reason is not None,
        reason=_skip_reason or "skipped",
    )

    @_skip_mark
    def test_graph_quality_golden_set() -> None:
        """Run golden-set evaluation and assert recall thresholds are met."""
        results = run_quality_check()

        # Print per-case summary for CI visibility
        print("\n── Graph Quality Evaluation ──────────────────────────────")
        for case in results["cases"]:
            status = "✓" if case["entity_recall"] >= ENTITY_RECALL_MIN else "✗"
            print(
                f"  {status} [{case['id']}] {case['description']}"
                f" | entity_recall={case['entity_recall']:.0%}"
                f" | relation_recall={case['relation_recall']:.0%}"
                f" | latency={case['latency_ms']}ms"
            )
            if case["entity_misses"]:
                print(f"    entity misses: {case['entity_misses']}")
            if case["relation_misses"]:
                print(f"    relation misses: {case['relation_misses']}")

        print(
            f"\n  Overall entity recall:   {results['overall_entity_recall']:.1%}"
            f" (threshold: {ENTITY_RECALL_MIN:.0%})"
        )
        print(
            f"  Overall relation recall: {results['overall_relation_recall']:.1%}"
            f" (threshold: {RELATION_RECALL_MIN:.0%})"
        )
        print("──────────────────────────────────────────────────────────\n")

        assert results["overall_entity_recall"] >= ENTITY_RECALL_MIN, (
            f"Entity recall {results['overall_entity_recall']:.1%} is below "
            f"threshold {ENTITY_RECALL_MIN:.0%}.\n"
            f"Failing cases:\n"
            + "\n".join(
                f"  [{c['id']}] entity_recall={c['entity_recall']:.0%}, "
                f"misses={c['entity_misses']}"
                for c in results["failures"]
            )
        )

        assert results["overall_relation_recall"] >= RELATION_RECALL_MIN, (
            f"Relation recall {results['overall_relation_recall']:.1%} is below "
            f"threshold {RELATION_RECALL_MIN:.0%}."
        )

except ImportError:
    # pytest not available — standalone mode only
    pass


# ── Standalone execution ───────────────────────────────────────────────────────

if __name__ == "__main__":
    if _skip_reason:
        print(f"SKIP: {_skip_reason}")
        sys.exit(0)

    print("Running graph quality golden-set evaluation…")
    results = run_quality_check()

    print(f"\nTest cases: {results['total_test_cases']}")
    print(
        f"Overall entity recall:   {results['overall_entity_recall']:.1%}"
        f" (threshold: {ENTITY_RECALL_MIN:.0%})"
    )
    print(
        f"Overall relation recall: {results['overall_relation_recall']:.1%}"
        f" (threshold: {RELATION_RECALL_MIN:.0%})"
    )
    print()

    for case in results["cases"]:
        ent_ok = "✓" if case["entity_recall"] >= ENTITY_RECALL_MIN else "✗"
        rel_ok = "✓" if case["relation_recall"] >= RELATION_RECALL_MIN else "✗"
        print(
            f"  {ent_ok}/{rel_ok} [{case['id']}] {case['description']}"
            f"  ent={case['entity_recall']:.0%}  rel={case['relation_recall']:.0%}"
            f"  latency={case['latency_ms']}ms"
        )
        if case["entity_misses"]:
            print(f"       entity misses : {case['entity_misses']}")
        if case["relation_misses"]:
            print(f"       relation misses: {case['relation_misses']}")

    print()
    if results["passed"]:
        print("✓ PASSED — all thresholds met.")
        sys.exit(0)
    else:
        print("✗ FAILED — one or more thresholds not met.")
        sys.exit(1)
