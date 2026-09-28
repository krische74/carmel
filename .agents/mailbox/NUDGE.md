# Nudge prompt: restart a stalled executor

Paste into the existing executor chat when it has stopped mid-run. It is shorter than
KICKOFF.md because the agent has already read the protocol.

---

Resume the mailbox loop. `.agents/mailbox/GUARDRAILS.md` still applies.

    .venv\Scripts\python .agents/mailbox/mbx.py watch --as cursor --timeout 240

If `watch` shows nothing but you had a task in progress, finish that task first.

- Exit 0: do the task, run the gate, post your reply with `mbx.py post --as cursor`, then
  run `watch` again.
- Exit 4: run `watch` again.
- Exit 3: the session is closed. Stop and say so.

Only exit 3 ends the loop. Reply with evidence by file path.
