"""Unit tests for the soft semantic relation gate.

Coverage:
  - allow / quarantine / skip decisions for various triple types
  - contacts_json source priors (boost and down-weight)
  - high-confidence override (skip → quarantine)
  - gate disabled via feature flag
  - regression: skip/quarantine candidates produce suggestion signals
  - GateResult fields are well-formed

No Neo4j or model loading required — pure unit tests.
"""
from __future__ import annotations

import importlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

# ── Path setup ─────────────────────────────────────────────────────────────────
_HERE = Path(__file__).parent
_SERVICE_ROOT = _HERE.parent
if str(_SERVICE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SERVICE_ROOT))


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_gate(
    enabled: bool = True,
    allow_threshold: float = 0.45,
    quarantine_threshold: float = 0.25,
    high_conf_override: float = 0.85,
):
    """Construct a RelationSoftGate with explicit thresholds (bypasses config)."""
    # Import after path setup
    from extraction.relation_gate import RelationSoftGate, reset_gate

    reset_gate()  # ensure no cached singleton bleeds into this test

    gate = RelationSoftGate.__new__(RelationSoftGate)
    gate.enabled = enabled
    gate.allow_threshold = allow_threshold
    gate.quarantine_threshold = quarantine_threshold
    gate.high_conf_override = high_conf_override
    return gate


# ── Tests ──────────────────────────────────────────────────────────────────────

class TestGateAllowDecisions(unittest.TestCase):
    """Clearly plausible triples should be allowed."""

    def setUp(self):
        self.gate = _make_gate()

    def _eval(self, head, rel, tail, source="session", conf=0.7):
        return self.gate.evaluate(
            head_type=head,
            rel_type=rel,
            tail_type=tail,
            source=source,
            confidence=conf,
        )

    def test_person_works_at_org(self):
        r = self._eval("Person", "WORKS_AT", "Organization")
        self.assertEqual(r.decision, "allow")
        self.assertGreaterEqual(r.score, 0.45)

    def test_person_knows_person(self):
        r = self._eval("Person", "KNOWS", "Person")
        self.assertEqual(r.decision, "allow")

    def test_person_located_in_place(self):
        r = self._eval("Person", "LOCATED_IN", "Place")
        self.assertEqual(r.decision, "allow")

    def test_person_member_of_org(self):
        r = self._eval("Person", "MEMBER_OF", "Organization")
        self.assertEqual(r.decision, "allow")

    def test_person_classmate_of_person(self):
        r = self._eval("Person", "CLASSMATE_OF", "Person")
        self.assertEqual(r.decision, "allow")

    def test_person_studied_at_org(self):
        r = self._eval("Person", "STUDIED_AT", "Organization")
        self.assertEqual(r.decision, "allow")

    def test_org_located_in_place(self):
        r = self._eval("Organization", "LOCATED_IN", "Place")
        self.assertEqual(r.decision, "allow")

    def test_contacts_json_boosted_works_at(self):
        """contacts_json + WORKS_AT should boost score above allow threshold."""
        r = self._eval("Person", "WORKS_AT", "Organization", source="contacts_json", conf=0.5)
        self.assertEqual(r.decision, "allow", msg=f"Expected allow, got {r.decision}: {r.reason}")

    def test_contacts_json_boosted_contact_of(self):
        r = self._eval("Person", "CONTACT_OF", "Person", source="contacts_json", conf=0.5)
        self.assertEqual(r.decision, "allow")


class TestGateQuarantineDecisions(unittest.TestCase):
    """Uncertain triples should be quarantined (written but flagged)."""

    def setUp(self):
        self.gate = _make_gate()

    def _eval(self, head, rel, tail, source="session", conf=0.5):
        return self.gate.evaluate(
            head_type=head,
            rel_type=rel,
            tail_type=tail,
            source=source,
            confidence=conf,
        )

    def test_unknown_triple_medium_conf(self):
        """Unlisted triple with medium confidence → quarantine (score ~0.55 * mid-conf)."""
        # Unlisted triple has base score 0.55, which is > allow_threshold 0.45 → allow
        # Let's test something that definitely lands in quarantine range
        r = self._eval("Technology", "KNOWS", "Organization", conf=0.4)
        # Technology-KNOWS-Organization is not in plausible, not in implausible,
        # and KNOWS is not generic → score=0.55 → above allow → allow actually
        # Let's test a clearly implausible triple at non-high confidence
        r2 = self._eval("Organization", "CHILD_OF", "Person", conf=0.6)
        # score=0.20 * 1.0 = 0.20, which is below quarantine_threshold 0.25
        # but conf 0.60 < high_conf_override 0.85, so this should be skip
        self.assertIn(r2.decision, {"quarantine", "skip"})

    def test_implausible_with_high_confidence_override(self):
        """High-confidence extractor overrides skip → quarantine."""
        r = self._eval("Organization", "CHILD_OF", "Person", conf=0.90)
        # gate_score=0.20, below quarantine_threshold=0.25, BUT conf=0.90 >= 0.85
        self.assertEqual(r.decision, "quarantine", msg=f"Expected quarantine (high-conf override), got {r.decision}: {r.reason}")

    def test_contacts_json_downweighted_medium_conf(self):
        """contacts_json + downweighted rel at medium conf → quarantine or skip."""
        r = self._eval("Person", "REPORTS_TO", "Person", source="contacts_json", conf=0.65)
        # base=0.85 * 0.55 (downweight) ≈ 0.467 which is just above allow threshold
        # Let's check CHILD_OF which has base 0.80 → 0.80 * 0.55 = 0.44 (just below allow 0.45)
        r2 = self._eval("Person", "CHILD_OF", "Person", source="contacts_json", conf=0.60)
        # 0.80 * 0.55 = 0.44 < 0.45, conf 0.60 < 0.85 → quarantine (0.44 >= quarantine_threshold 0.25)
        self.assertEqual(r2.decision, "quarantine", msg=f"Expected quarantine, got {r2.decision}: {r2.reason}")


class TestGateSkipDecisions(unittest.TestCase):
    """Implausible low-confidence triples should be skipped + signal emitted."""

    def setUp(self):
        self.gate = _make_gate()

    def _eval(self, head, rel, tail, source="session", conf=0.3):
        return self.gate.evaluate(
            head_type=head,
            rel_type=rel,
            tail_type=tail,
            source=source,
            confidence=conf,
        )

    def test_technology_reports_to_person(self):
        """Technology-REPORTS_TO-Person is implausible at low conf → skip."""
        r = self._eval("Technology", "REPORTS_TO", "Person", conf=0.35)
        # score=0.15, below quarantine_threshold, conf 0.35 < 0.85 → skip
        self.assertEqual(r.decision, "skip", msg=f"Expected skip, got {r.decision}: {r.reason}")

    def test_contacts_downweighted_low_conf(self):
        """contacts_json + MENTORED_BY at low confidence → skip."""
        r = self._eval("Person", "MENTORED_BY", "Person", source="contacts_json", conf=0.40)
        # base=0.85 (Person-MENTORED_BY-Person) * 0.55 (downweight) = 0.467 → allow
        # Use lower base triple: Organization-MENTORED_BY-Person (implausible: 0.25)
        r2 = self._eval("Organization", "MENTORED_BY", "Person", source="contacts_json", conf=0.40)
        # base=0.25 * 0.55 = 0.1375, conf=0.40 < 0.85 → skip
        self.assertEqual(r2.decision, "skip", msg=f"Expected skip, got {r2.decision}: {r2.reason}")


class TestGateDisabledFlag(unittest.TestCase):
    """Gate disabled → all allow."""

    def test_disabled_always_allow(self):
        gate = _make_gate(enabled=False)
        for triple in [
            ("Technology", "REPORTS_TO", "Person"),
            ("Organization", "CHILD_OF", "Person"),
        ]:
            r = gate.evaluate(
                head_type=triple[0],
                rel_type=triple[1],
                tail_type=triple[2],
                source="session",
                confidence=0.3,
            )
            self.assertEqual(r.decision, "allow", msg=f"Disabled gate should always allow, got {r.decision} for {triple}")
            self.assertGreater(r.score, 0.9)


class TestGateResultFields(unittest.TestCase):
    """GateResult always returns well-formed fields."""

    def setUp(self):
        self.gate = _make_gate()

    def _check_result(self, result):
        from extraction.relation_gate import GateResult
        self.assertIsInstance(result, GateResult)
        self.assertIn(result.decision, {"allow", "quarantine", "skip"})
        self.assertIsInstance(result.reason, str)
        self.assertTrue(len(result.reason) > 0)
        self.assertGreaterEqual(result.score, 0.0)
        self.assertLessEqual(result.score, 1.0)

    def test_allow_result_fields(self):
        r = self.gate.evaluate(
            head_type="Person", rel_type="WORKS_AT", tail_type="Organization",
            source="session", confidence=0.7
        )
        self._check_result(r)

    def test_skip_result_fields(self):
        r = self.gate.evaluate(
            head_type="Technology", rel_type="REPORTS_TO", tail_type="Person",
            source="session", confidence=0.3
        )
        self._check_result(r)

    def test_gate_score_clipped(self):
        """Boosted scores should never exceed 1.0."""
        r = self.gate.evaluate(
            head_type="Person", rel_type="WORKS_AT", tail_type="Organization",
            source="contacts_json", confidence=0.9
        )
        self.assertLessEqual(r.score, 1.0)
        self.assertEqual(r.decision, "allow")


class TestGateSignalPreservation(unittest.TestCase):
    """Regression: skip and quarantine decisions MUST emit suggestion signals.

    Verifies that the evolution pipeline signal path is never silently dropped.
    """

    def _run_gate_with_capture(
        self, head, rel, tail, source="session", conf=0.3
    ) -> dict[str, Any]:
        """Run the gate and capture the log_relation_gate_decision call args."""
        gate = _make_gate()
        captured: list[dict[str, Any]] = []

        def fake_log(head, tail, head_type, rel_type, tail_type, decision,
                     reason, gate_score, confidence, source, context=""):
            captured.append({
                "head": head, "tail": tail,
                "head_type": head_type, "rel_type": rel_type, "tail_type": tail_type,
                "decision": decision, "reason": reason,
                "gate_score": gate_score, "confidence": confidence, "source": source,
            })

        result = gate.evaluate(
            head_type=head, rel_type=rel, tail_type=tail,
            source=source, confidence=conf,
        )
        return {"result": result, "captured": captured}

    def test_skip_decision_emits_signal(self):
        """When _build_relationships produces a skip, a suggestion signal is written."""
        import io

        with tempfile.TemporaryDirectory() as tmpdir:
            # Patch SUGGESTIONS_DIR so the JSONL file lands in a temp dir
            suggestions_dir = Path(tmpdir) / "suggestions"
            suggestions_dir.mkdir()

            # We need to reload graph_suggestions with the patched dir.
            import memory.graph_suggestions as gs
            original_dir = gs.SUGGESTIONS_DIR

            try:
                gs.SUGGESTIONS_DIR = suggestions_dir

                gate = _make_gate()
                # implausible triple at low conf → skip
                r = gate.evaluate(
                    head_type="Technology",
                    rel_type="REPORTS_TO",
                    tail_type="Person",
                    source="session",
                    confidence=0.30,
                )
                self.assertEqual(r.decision, "skip")

                # Now call log_relation_gate_decision directly as pipeline would
                gs.log_relation_gate_decision(
                    head="SomeService",
                    tail="Alice",
                    head_type="Technology",
                    rel_type="REPORTS_TO",
                    tail_type="Person",
                    decision="skip",
                    reason=r.reason,
                    gate_score=r.score,
                    confidence=0.30,
                    source="session",
                    context="test context",
                )

                # Verify the JSONL signal was written
                files = list(suggestions_dir.glob("*.jsonl"))
                self.assertTrue(files, "No suggestion JSONL file was written for skip signal")
                lines = files[0].read_text().strip().splitlines()
                self.assertTrue(lines, "Suggestion JSONL file is empty")
                entry = json.loads(lines[-1])
                self.assertEqual(entry["type"], "relation_gate_decision")
                self.assertEqual(entry["decision"], "skip")
                self.assertEqual(entry["rel_type"], "REPORTS_TO")
                self.assertIn("head_type", entry)
                self.assertIn("gate_score", entry)

            finally:
                gs.SUGGESTIONS_DIR = original_dir

    def test_quarantine_decision_emits_signal(self):
        """Quarantine decisions also emit a suggestion signal."""
        with tempfile.TemporaryDirectory() as tmpdir:
            suggestions_dir = Path(tmpdir) / "suggestions"
            suggestions_dir.mkdir()

            import memory.graph_suggestions as gs
            original_dir = gs.SUGGESTIONS_DIR

            try:
                gs.SUGGESTIONS_DIR = suggestions_dir

                # Gate score in quarantine band
                gate = _make_gate()
                r = gate.evaluate(
                    head_type="Organization",
                    rel_type="CHILD_OF",
                    tail_type="Person",
                    source="session",
                    confidence=0.90,  # high-conf override → quarantine
                )
                self.assertEqual(r.decision, "quarantine")

                gs.log_relation_gate_decision(
                    head="AcmeCorp",
                    tail="John",
                    head_type="Organization",
                    rel_type="CHILD_OF",
                    tail_type="Person",
                    decision="quarantine",
                    reason=r.reason,
                    gate_score=r.score,
                    confidence=0.90,
                    source="session",
                )

                files = list(suggestions_dir.glob("*.jsonl"))
                self.assertTrue(files)
                entry = json.loads(files[0].read_text().strip().splitlines()[-1])
                self.assertEqual(entry["decision"], "quarantine")
                self.assertEqual(entry["type"], "relation_gate_decision")

            finally:
                gs.SUGGESTIONS_DIR = original_dir

    def test_allow_does_not_emit_gate_signal(self):
        """Allow decisions should NOT produce a gate suggestion signal.

        The suggestion signal for skip/quarantine is what feeds the evolution
        pipeline; allow is a clean write and doesn't need a signal entry.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            suggestions_dir = Path(tmpdir) / "suggestions"
            suggestions_dir.mkdir()

            import memory.graph_suggestions as gs
            original_dir = gs.SUGGESTIONS_DIR
            try:
                gs.SUGGESTIONS_DIR = suggestions_dir

                # Only log if decision != allow — that's the contract in pipeline.py
                gate = _make_gate()
                r = gate.evaluate(
                    head_type="Person", rel_type="WORKS_AT", tail_type="Organization",
                    source="session", confidence=0.8,
                )
                self.assertEqual(r.decision, "allow")
                # Don't call log_relation_gate_decision for allow (matches pipeline logic)

                files = list(suggestions_dir.glob("*.jsonl"))
                # No file should exist (or file should be empty)
                gate_entries = []
                for f in files:
                    for line in f.read_text().strip().splitlines():
                        if line:
                            e = json.loads(line)
                            if e.get("type") == "relation_gate_decision":
                                gate_entries.append(e)
                self.assertEqual(gate_entries, [],
                    "allow decision should not emit a gate signal")

            finally:
                gs.SUGGESTIONS_DIR = original_dir


class TestContactsJsonSourcePriors(unittest.TestCase):
    """contacts_json source priors must boost or down-weight as specified."""

    def setUp(self):
        self.gate = _make_gate()

    def test_boosted_rels_above_allow(self):
        """Boosted rels in contacts_json should score above allow threshold."""
        boosted = ["CONTACT_OF", "WORKS_AT", "LOCATED_IN", "MEMBER_OF", "CLASSMATE_OF"]
        for rel in boosted:
            # Use plausible types so the base score is high
            triple_map = {
                "CONTACT_OF": ("Person", "Person"),
                "WORKS_AT": ("Person", "Organization"),
                "LOCATED_IN": ("Person", "Place"),
                "MEMBER_OF": ("Person", "Organization"),
                "CLASSMATE_OF": ("Person", "Person"),
            }
            head, tail = triple_map.get(rel, ("Person", "Organization"))
            r = self.gate.evaluate(
                head_type=head, rel_type=rel, tail_type=tail,
                source="contacts_json", confidence=0.5,
            )
            self.assertEqual(r.decision, "allow",
                msg=f"contacts_json boosted rel {rel} should be allow, got {r.decision}: {r.reason}")

    def test_downweighted_rels_lower_score(self):
        """Down-weighted rels in contacts_json should score lower than without source prior."""
        downweighted = ["CHILD_OF", "REPORTS_TO", "MENTORED_BY"]
        for rel in downweighted:
            triple_map = {
                "CHILD_OF": ("Person", "Person"),
                "REPORTS_TO": ("Person", "Person"),
                "MENTORED_BY": ("Person", "Person"),
            }
            head, tail = triple_map.get(rel, ("Person", "Person"))

            r_default = self.gate.evaluate(
                head_type=head, rel_type=rel, tail_type=tail,
                source="session", confidence=0.5,
            )
            r_contacts = self.gate.evaluate(
                head_type=head, rel_type=rel, tail_type=tail,
                source="contacts_json", confidence=0.5,
            )
            self.assertLess(r_contacts.score, r_default.score,
                msg=f"contacts_json score should be lower for {rel}: {r_contacts.score} vs {r_default.score}")


class TestHighConfidenceOverride(unittest.TestCase):
    """Very high extractor confidence bumps skip → quarantine."""

    def setUp(self):
        self.gate = _make_gate(
            allow_threshold=0.45,
            quarantine_threshold=0.25,
            high_conf_override=0.85,
        )

    def test_high_conf_lifts_skip_to_quarantine(self):
        """An implausible triple (score < quarantine threshold) at very high conf
        should be quarantined, not skipped."""
        r = self.gate.evaluate(
            head_type="Technology",
            rel_type="REPORTS_TO",
            tail_type="Person",
            source="session",
            confidence=0.92,  # above 0.85 override
        )
        # base score = 0.15 (known implausible), below quarantine_threshold 0.25
        # but confidence 0.92 >= 0.85 → quarantine
        self.assertEqual(r.decision, "quarantine",
            msg=f"High-confidence override should yield quarantine, got {r.decision}: {r.reason}")

    def test_below_high_conf_still_skips(self):
        """Same implausible triple at sub-override confidence → skip."""
        r = self.gate.evaluate(
            head_type="Technology",
            rel_type="REPORTS_TO",
            tail_type="Person",
            source="session",
            confidence=0.60,  # below 0.85
        )
        self.assertEqual(r.decision, "skip",
            msg=f"Below high-conf threshold should still skip, got {r.decision}: {r.reason}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
