#!/usr/bin/env python
"""Cursor `stop` hook — auto-continues the agent when Woebbe posts a new task.

Fires when the Cursor agent finishes a turn. Behaviour:
  * mailbox session not armed (halted / HALT file / cap / deadline) -> {}  (agent stops normally)
  * new message waiting or arriving within WAIT seconds             -> {"followup_message": ...}
  * nothing arrives before WAIT                                     -> cheap re-arm follow-up

The halted flag is the master switch: when no session is armed this hook is a no-op,
so normal interactive Cursor work is never hijacked.

Wire up in .cursor/hooks.json with a timeout slightly greater than WAIT.
"""
from __future__ import annotations

import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import mbx  # noqa: E402

WAIT = float(os.environ.get("MAILBOX_HOOK_WAIT", "570"))
POLL = 2.0

PREAMBLE = """[AUTOMATED HANDOFF — Carmel agent mailbox]

A new message from Woebbe is below. Governance in AGENTS.md still applies in full: TDD,
ruff clean, no scope creep, evidence over assumption, `git add .` banned.

When you have finished (or are blocked), reply through the mailbox — do NOT just answer in chat:

    .venv\\Scripts\\python .agents/mailbox/mbx.py post --as cursor --status DONE --subject "<one line>" --body-file <your_reply.md>

Use --status BLOCKED instead if you need a decision. Your reply must cite evidence by file path
(test output, JSON artifacts, diffs) rather than asserting results in prose. Then stop; the hook
will bring you Woebbe's next message automatically.

--- MESSAGE ---
"""

REARM = """[AUTOMATED HANDOFF — no new task yet]

Woebbe has not posted a NEW mailbox message yet and the session is still open.

FIRST: if a previously delivered task or an operator-authorized contract arm (e.g. a numbered
tier arm green-lit in chat) is unfinished, CONTINUE THAT WORK NOW — this re-arm notice never
cancels or defers authorized in-progress work; it only means no additional message has arrived.

Only if you have nothing authorized and unfinished, do not start new work. Run exactly this and
follow its output:

    .venv\\Scripts\\python .agents/mailbox/mbx.py watch --as cursor --timeout 240

Exit 0 = a new task is printed, act on it. Exit 4 = still nothing, just stop and say
"waiting on Woebbe". Exit 3 = the session is closed, stop and say so.
"""


HOOK_LOG = os.path.join(HERE, "log", "hook.log")


def hook_log(line: str) -> None:
    """Append one line per event so 'did the hook fire?' is a file read, not a guess
    (Fable review A-5). Never let logging break the hook itself."""
    try:
        os.makedirs(os.path.dirname(HOOK_LOG), exist_ok=True)
        with open(HOOK_LOG, "a", encoding="utf-8", newline="\n") as fh:
            fh.write(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} pid={os.getpid()} {line}\n")
    except OSError:
        pass


def emit(obj: dict) -> None:
    sys.stdout.write(json.dumps(obj))
    sys.stdout.flush()


def main() -> int:
    hook_log("invoked")
    try:
        payload = json.loads(sys.stdin.read() or "{}")
    except (json.JSONDecodeError, ValueError):
        payload = {}

    if payload.get("status") == "aborted":
        hook_log("exit: agent aborted, no follow-up")
        emit({})
        return 0

    reason = mbx.closed_reason(mbx.read_state())
    if reason:
        hook_log(f"exit: session closed ({reason}), no follow-up")
        emit({})
        return 0

    path = mbx.inbox("cursor")
    deadline = time.time() + WAIT
    while True:
        state = mbx.read_state()
        reason = mbx.closed_reason(state)
        msg = mbx.parse(path)
        seen = int(state.get("seen", {}).get("cursor", 0))
        if msg and msg["seq"] > seen:
            def mark(s, _seq=msg["seq"]):
                s.setdefault("seen", {})["cursor"] = _seq
                return None

            mbx.update_state(mark)
            hook_log(f"exit: delivered seq={msg['seq']} as follow-up")
            emit({"followup_message": PREAMBLE + msg["raw"].rstrip("\n")})
            return 0
        if reason:
            hook_log(f"exit: session closed mid-wait ({reason}), no follow-up")
            emit({})
            return 0
        if time.time() >= deadline:
            hook_log(f"exit: waited {WAIT}s, nothing arrived, emitting re-arm")
            emit({"followup_message": REARM})
            return 0
        time.sleep(POLL)


if __name__ == "__main__":
    raise SystemExit(main())
