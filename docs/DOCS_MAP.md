# MollyGraph Docs Map

This file tells you which markdown files are current and which plans are still intentionally kept around.

## Authoritative now

- `README.md`
  Repo-level product story and default stack.
- `docs/ARCHITECTURE.md`
  Current architecture summary.
- `service/.env.example`
  Current runtime configuration surface.
- `service/README.md`
  Service-level runtime summary.
- `service/BACKLOG.md`
  Service-focused backlog summary aligned with the current refactor.
- `sdk/README.md`
  Current SDK and MCP adapter usage.

## Experimental plans

- `service/DECISION_TRACES_PLAN.md`
  Later-phase differentiator, not part of the default runtime.

## Removed historical docs

The older Neo4j-era audit and gap documents were removed during the local-first cleanup to keep the repo surface smaller and less confusing. If you need them, use git history.

## Rule of thumb

If a markdown file still assumes Neo4j is the default backend, or treats audit/training/LoRA loops as the base product, it should either be updated immediately or removed.
