
### Debug Local/Ollama Models for Audit (Local-First Goal)
- **Priority:** Medium
- **Goal:** Get ollama (qwen3:4b or better) producing parseable JSON audit verdicts
- **Why:** Reduce dependency on Moonshot API, lower latency, zero cost
- **Issues:** qwen3:4b returns unparseable output for structured JSON batch verdicts
- **Options:** 
  - Try larger ollama model (qwen3:8b, llama3.1:8b)
  - Add JSON mode/grammar constraints to ollama call
  - Fine-tune a small model on audit verdict examples
  - Use ollama structured output (json_schema parameter)
- **Current workaround:** Ollama tier skipped, using Moonshot primary
