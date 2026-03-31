# Production Test Checklist

Use this checklist before calling the current Ladybug-first runtime ready for a production-style test pass.

## Blocking checks

Confirm the runtime is on Python `3.12`:

```bash
service/.venv/bin/python --version
```

That command should report `Python 3.12.x`.

Run the full automated suite:

```bash
service/.venv/bin/python -m pytest -q
```

Run the isolated local-runtime smoke runner:

```bash
service/.venv/bin/python scripts/production_smoke.py --json
```

Run the opt-in pytest wrapper for the real subprocess smoke path:

```bash
MOLLYGRAPH_RUN_RUNTIME_SMOKE=1 service/.venv/bin/python -m pytest tests/test_production_smoke.py -m runtime -q
```

## Expected pass conditions

- the full suite is green
- the smoke runner reports `"status": "ok"`
- startup health is `healthy`
- a fresh runtime does not report unexpected `operator_advisories`
- the first ingest completes and retrieval works
- no stray `uvicorn main:app` process is left running after the smoke pass

## Informational, not blocking

- the first ingest may still take a few seconds while local models warm up
- `operator_advisories` are expected on non-fresh runtimes that already have queue activity
- concurrent local model-heavy runs should now warn clearly instead of silently stacking load

## Cleanup check

If you want to confirm no test service is still running:

```bash
ps -Ao pid,etime,command | egrep 'uvicorn main:app' | egrep -v 'egrep'
```

That command should return no MollyGraph test service unless you intentionally left one running.
