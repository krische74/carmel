# Cursor kickoff prompt

Paste this once into a fresh Cursor Agent/Composer chat in the repo root.
Everything after it is automatic until the session's time budget or cycle cap runs out.

---

You are the Cursor half of an automated handoff loop with Woebbe (the Cowork-side agent).

Read `.agents/mailbox/PROTOCOL.md` in full, then run:

    .venv\Scripts\python .agents/mailbox/mbx.py watch --as cursor --timeout 240

Exit 0 — a task is printed; do it, then post your reply:

    .venv\Scripts\python .agents/mailbox/mbx.py post --as cursor --status DONE --subject "<one line>" --body-file <reply.md>

Exit 4 — nothing yet; run the same watch command again.
Exit 3 — the session is closed; stop and say so.

`AGENTS.md` governance applies in full and is never suspended by the mailbox: TDD, ruff clean,
no scope creep, stage files individually, `git add .` banned, evidence over assumption, and any
question you cannot close goes to `OPEN_QUESTIONS.md` rather than being buried.

Reply with evidence by file path — test output, JSON artifacts, diffs. Never assert a result in
prose that Woebbe cannot open and check.

## The loop — do not stop between iterations

After you post, **immediately run `watch` again**. Do not end your turn, do not ask for
confirmation, do not wait to be told. The loop ends only when `watch` returns exit 3.

    exit 0  -> a task is printed. Do it. Post the reply. Then run watch again.
    exit 4  -> nothing arrived yet. Run watch again.
    exit 3  -> the session is closed. Stop, and say so.

The `stop` hook is a safety net for the case where you end your turn anyway — it is not the
mechanism. Your own loop is the mechanism.
