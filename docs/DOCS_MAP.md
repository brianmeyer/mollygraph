# MollyGraph Docs Map

This file tells you which markdown files are current and which ones are optional planning references.

## Authoritative now

- `README.md`
  Repo-level product story and default stack.
- `docs/ARCHITECTURE.md`
  Current architecture summary.
- `service/.env.example`
  Current runtime configuration surface.
- `service/README.md`
  Service-level runtime summary.
- `docs/PRODUCTION_TEST_CHECKLIST.md`
  Final pre-production test pass for the local-first runtime.
- `service/BACKLOG.md`
  Service-focused backlog summary for the current product.
- `sdk/README.md`
  Current SDK and MCP adapter usage.

## Experimental plans

- `service/DECISION_TRACES_PLAN.md`
  Later-phase differentiator, not part of the default runtime.

## Historical note

Older Neo4j-era audit and gap documents were removed during cleanup. If you need them, use git history.

## Rule of thumb

If a markdown file assumes Neo4j is the default backend, or treats audit/training/LoRA loops as the base product, it should be updated or removed.
