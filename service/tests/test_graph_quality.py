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
from collections import defaultdict
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


def _clean_text(value: str) -> str:
    return re.sub(r"[^\w\s]", "", value).strip().lower()


def _entity_texts(extracted_entities: list[dict[str, Any]]) -> list[str]:
    return [
        str(ent.get("text") or "")
        for ent in extracted_entities
        if str(ent.get("text") or "").strip()
    ]


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
    src_clean = _clean_text(expected_source)
    tgt_clean = _clean_text(expected_target)

    for rel in extracted_relations:
        head = _clean_text(str(rel.get("head") or ""))
        tail = _clean_text(str(rel.get("tail") or ""))
        label = _normalize_rel_type(str(rel.get("label") or ""))

        head_match = src_clean in head or head in src_clean
        tail_match = tgt_clean in tail or tail in tgt_clean
        type_match = bool(pattern.fullmatch(label))

        if head_match and tail_match and type_match:
            return True
    return False


def _matches_forbidden_entity(forbidden: str, extracted_entities: list[dict[str, Any]]) -> str | None:
    needle = _clean_text(forbidden)
    if not needle:
        return None
    for extracted in _entity_texts(extracted_entities):
        hay = _clean_text(extracted)
        if hay and (needle in hay or hay in needle):
            return extracted
    return None


def _case_category(tc: dict[str, Any]) -> str:
    return str(tc.get("category") or "baseline").strip().lower()


def _is_known_gap(tc: dict[str, Any]) -> bool:
    return bool(tc.get("known_gap", False))


def _format_case_issue(case: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    if case.get("entity_recall", 1.0) < ENTITY_RECALL_MIN:
        issues.append(f"entity recall {case['entity_recall']:.0%} < {ENTITY_RECALL_MIN:.0%}")
    if case.get("relation_recall", 1.0) < RELATION_RECALL_MIN:
        issues.append(f"relation recall {case['relation_recall']:.0%} < {RELATION_RECALL_MIN:.0%}")
    if case.get("forbidden_entity_hits"):
        issues.append(f"forbidden entities {case['forbidden_entity_hits']}")
    if case.get("forbidden_relation_hits"):
        issues.append(f"forbidden relations {case['forbidden_relation_hits']}")
    return issues


def _render_report(results: dict[str, Any]) -> str:
    lines = ["── Graph Quality Evaluation ──────────────────────────────"]
    for case in results["cases"]:
        if case["passed"]:
            status = "✓"
        elif case.get("known_gap"):
            status = "!"
        else:
            status = "✗"
        lines.append(
            f"  {status} [{case['id']}] {case['category']} :: {case['description']}"
            f" | entity_recall={case['entity_recall']:.0%}"
            f" | relation_recall={case['relation_recall']:.0%}"
            f" | latency={case['latency_ms']}ms"
        )
        if case["entity_misses"]:
            lines.append(f"    entity misses: {case['entity_misses']}")
        if case["relation_misses"]:
            lines.append(f"    relation misses: {case['relation_misses']}")
        if case["forbidden_entity_hits"]:
            lines.append(f"    forbidden entity hits: {case['forbidden_entity_hits']}")
        if case["forbidden_relation_hits"]:
            lines.append(f"    forbidden relation hits: {case['forbidden_relation_hits']}")

    lines.append("")
    lines.append(
        f"  Overall entity recall:   {results['overall_entity_recall']:.1%}"
        f" (threshold: {ENTITY_RECALL_MIN:.0%})"
    )
    lines.append(
        f"  Overall relation recall: {results['overall_relation_recall']:.1%}"
        f" (threshold: {RELATION_RECALL_MIN:.0%})"
    )
    if results.get("category_summary"):
        lines.append("  By category:")
        for category, summary in sorted(results["category_summary"].items()):
            lines.append(
                f"    {category}: {summary['passed']}/{summary['cases']} passed"
                f" | ent={summary['entity_recall']:.0%}"
                f" | rel={summary['relation_recall']:.0%}"
            )
    lines.append("──────────────────────────────────────────────────────────")
    return "\n".join(lines)


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
        category_summary: dict[str, dict[str, int]] = defaultdict(
            lambda: {
                "cases": 0,
                "passed": 0,
                "entities_expected": 0,
                "entities_found": 0,
                "relations_expected": 0,
                "relations_found": 0,
            }
        )

        total_entities_expected = 0
        total_entities_found = 0
        total_relations_expected = 0
        total_relations_found = 0

        for tc in golden:
            tc_id = tc.get("id", "?")
            text = str(tc.get("input", ""))
            expected_entities = tc.get("expected_entities", [])
            expected_relations = tc.get("expected_relations", [])
            category = _case_category(tc)

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

            forbidden_entity_hits: list[str] = []
            for forbidden in tc.get("forbidden_entities", []):
                hit = _matches_forbidden_entity(str(forbidden), extracted_entities)
                if hit:
                    forbidden_entity_hits.append(hit)

            forbidden_relation_hits: list[str] = []
            for forbidden in tc.get("forbidden_relations", []):
                if _source_target_match(
                    str(forbidden["source"]),
                    str(forbidden["target"]),
                    extracted_relations,
                    str(forbidden["type_pattern"]),
                ):
                    forbidden_relation_hits.append(
                        f"{forbidden['source']} --[{forbidden['type_pattern']}]--> {forbidden['target']}"
                    )

            n_exp_rel = len(expected_relations)
            n_hit_rel = len(relation_hits)
            relation_recall = n_hit_rel / n_exp_rel if n_exp_rel > 0 else 1.0

            total_entities_expected += n_exp_ent
            total_entities_found += n_hit_ent
            total_relations_expected += n_exp_rel
            total_relations_found += n_hit_rel

            passed_case = (
                entity_recall >= ENTITY_RECALL_MIN
                and relation_recall >= RELATION_RECALL_MIN
                and not forbidden_entity_hits
                and not forbidden_relation_hits
            )

            category_summary[category]["cases"] += 1
            category_summary[category]["entities_expected"] += n_exp_ent
            category_summary[category]["entities_found"] += n_hit_ent
            category_summary[category]["relations_expected"] += n_exp_rel
            category_summary[category]["relations_found"] += n_hit_rel
            if passed_case:
                category_summary[category]["passed"] += 1

            case_results.append(
                {
                    "id": tc_id,
                    "category": category,
                    "known_gap": _is_known_gap(tc),
                    "description": tc.get("description", ""),
                    "input": text,
                    "entity_recall": round(entity_recall, 3),
                    "entity_precision": round(entity_precision, 3),
                    "relation_recall": round(relation_recall, 3),
                    "passed": passed_case,
                    "entities_expected": n_exp_ent,
                    "entities_found": n_hit_ent,
                    "entity_misses": entity_misses,
                    "relations_expected": n_exp_rel,
                    "relations_found": n_hit_rel,
                    "relation_misses": [
                        f"{r['source']} --[{r['type_pattern']}]--> {r['target']}"
                        for r in relation_misses
                    ],
                    "forbidden_entity_hits": forbidden_entity_hits,
                    "forbidden_relation_hits": forbidden_relation_hits,
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
            and all(c["passed"] or c.get("known_gap") for c in case_results)
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
            "category_summary": {
                category: {
                    "cases": summary["cases"],
                    "passed": summary["passed"],
                    "entity_recall": round(
                        summary["entities_found"] / summary["entities_expected"]
                        if summary["entities_expected"] > 0
                        else 1.0,
                        4,
                    ),
                    "relation_recall": round(
                        summary["relations_found"] / summary["relations_expected"]
                        if summary["relations_expected"] > 0
                        else 1.0,
                        4,
                    ),
                }
                for category, summary in category_summary.items()
            },
            "failures": [c for c in case_results if not c["passed"] and not c.get("known_gap")],
            "known_gaps": [c for c in case_results if not c["passed"] and c.get("known_gap")],
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
        report = _render_report(results)
        print(f"\n{report}\n")

        assert results["passed"], (
            "Graph quality evaluation failed.\n"
            f"Overall entity recall: {results['overall_entity_recall']:.1%} "
            f"(threshold {ENTITY_RECALL_MIN:.0%})\n"
            f"Overall relation recall: {results['overall_relation_recall']:.1%} "
            f"(threshold {RELATION_RECALL_MIN:.0%})\n"
            + "\n".join(
                f"  [{c['id']}] {c['category']} :: {c['description']} -> "
                + "; ".join(_format_case_issue(c))
                for c in results["failures"]
            )
        )
        if results.get("known_gaps"):
            print(
                "Known graph-quality gaps still present:\n"
                + "\n".join(
                    f"  [{c['id']}] {c['category']} :: {c['description']} -> "
                    + "; ".join(_format_case_issue(c))
                    for c in results["known_gaps"]
                )
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
    print(_render_report(results))
    print()
    if results["passed"]:
        print("✓ PASSED — all thresholds met.")
        sys.exit(0)
    else:
        print("✗ FAILED — one or more thresholds not met.")
        sys.exit(1)
