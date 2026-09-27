# Claude Code guidance for Carmel

**Read [AGENTS.md](AGENTS.md) first** — it's the authoritative co-development guide (tech stack, non-negotiable rules, repository layout, testing discipline). Everything below is additive.

AGENTS.md now carries an **Open Questions Protocol**: any question you have gets raised in the response and routed to whatever can answer it (code/data → test → experiment → deep research → Keith), and anything that cannot close in-task goes to [OPEN_QUESTIONS.md](OPEN_QUESTIONS.md). "Worth revisiting later" is not a resolution.

## Trading System Context

- This is the Carmel trading system — paper mode primary, ETF-only MVP scope.
- Always verify idempotency keys are present on order submissions.
- When diagnosing log-based alerts, check if log contamination from pytest / unittest.mock is the source before blaming production code.
- Reconciliation warnings are a known minor issue — don't chase unless the user asks.

## Shell Environment

- Default shell is PowerShell on Windows — avoid bare `curl` with complex flags; use `Invoke-WebRequest` or `curl.exe` explicitly.
- After modifying Python entry points (e.g. the `carmel` CLI in `pyproject.toml`), remind the user to run `pip install -e .` to re-register console scripts.

## Code Review Expectations

- Perform thorough review before applying config or sizing changes — catch double-scaling, unit mismatches, and missing guardrails.
- Verify percent-of-equity and DCA math end-to-end when touching position sizing (regression tests must cover non–`risk_on` regimes).
