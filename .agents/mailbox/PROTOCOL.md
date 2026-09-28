# Agent mailbox protocol

A file-based channel that lets a planner agent (**woebbe**) and an executor agent
(**cursor**) hand work back and forth in this repository without a person copying prompts
between them. The planner writes tasks. The executor does them and replies with evidence. The
planner checks the evidence and writes the next task, or stops.

Read `GUARDRAILS.md` in this folder before anything else. It applies to every session.

## Layout

    .agents/mailbox/
      mbx.py            CLI: post / watch / session / status (both sides use it)
      stop_hook.py      Cursor `stop` hook: auto-continues the executor when a task lands
      guard_hook.py     Cursor `beforeShellExecution` / `beforeReadFile` hook: enforced guardrails
      config.json       role names, interpreter, gate commands, limits, guard rules
      GUARDRAILS.md     the rules
      PROTOCOL.md       this file
      KICKOFF.md        paste once into a fresh executor chat to start the loop
      NUDGE.md          paste into a stalled executor chat to restart it
      to_woebbe.md / to_cursor.md   inboxes (runtime, gitignored)
      state.json        seq / seen / cycle / deadline / halted (runtime, gitignored)
      HALT              first line not starting with `cleared` = hard stop (runtime, gitignored)
      log/              every posted message, numbered, plus hook.log and guard.log (runtime, gitignored)

## Message format

Each inbox holds one message, the newest. History lives in `log/`.

    <!-- MAILBOX
    seq: 7
    from: woebbe
    to: cursor
    status: READY
    subject: Add retry to the fetch client
    cycle: 3
    ts: 2026-09-27T13:10:00Z
    -->

    ...markdown body...

`seq` is one counter across both directions, so it also gives a total order. Writes go to a
temporary file and are then renamed into place, so a reader never sees half a message. A reader
acts only on `seq` greater than the last one it has seen.

**Statuses.** The planner posts `READY` (a task), `NOTE` or `ACK`. The executor posts `DONE`
(finished, evidence attached), `BLOCKED` (needs a decision), `ACK` or `NOTE`. Each side is
refused the other side's statuses. One cycle is one executor `DONE` or `BLOCKED`.

## Commands

    # arm a run (planner only). Both limits are required and capped by config.json.
    .venv\Scripts\python .agents/mailbox/mbx.py session start --minutes 90 --max-cycles 12

    # stop at once, from anywhere
    .venv\Scripts\python .agents/mailbox/mbx.py session stop

    .venv\Scripts\python .agents/mailbox/mbx.py status
    .venv\Scripts\python .agents/mailbox/mbx.py post  --as woebbe --status READY --subject "..." --body-file task.md
    .venv\Scripts\python .agents/mailbox/mbx.py post  --as cursor --status DONE --subject "..." --body-file reply.md
    .venv\Scripts\python .agents/mailbox/mbx.py watch --as cursor --timeout 240

`watch` exit codes: **0** new message printed, **3** session closed, **4** timed out with nothing new.

## When a session ends

A run ends when the first of these happens: the time limit passes, the cycle limit is reached,
`session stop` runs, or the first line of `HALT` stops starting with `cleared`. To stop by hand:

    echo halted > .agents/mailbox/HALT            (bash)
    Set-Content .agents/mailbox/HALT halted       (PowerShell)

`touch HALT` does not work: after `session start` the file exists with `cleared` on its first
line, and touch only updates the timestamp. `session start` rewrites HALT to `cleared`, so arming
a run clears a standing stop.

Closure is enforced by the transport. `watch --as cursor` checks closure before it delivers
anything, so a closed session never hands the executor new work, even if a message is waiting.
`watch --as woebbe` may still collect one final pending reply. A `READY` post into a closed
session is refused. `DONE`, `BLOCKED`, `ACK` and `NOTE` still post, with a warning, so finished
work is not lost to a deadline that passed mid-task. Once closed, the stop hook returns nothing
and the executor stays stopped until the planner arms the next run.

The kit fails closed. A session marked running with no cycle cap, no deadline, or a deadline it
cannot read counts as closed. An unreadable `state.json` or `HALT` counts as halted.

## How the executor gets woken

1. **The executor's own loop.** `KICKOFF.md` tells it to post, then run `watch` again, until
   `watch` exits 3. This is the main mechanism.
2. **The `stop` hook** (`.cursor/hooks.json`). When the agent ends a turn anyway, the hook waits
   up to about 9.5 minutes for a message and hands it back as the next turn. It does nothing
   while no session is armed. Its `loop_limit` is set from your limits as a backstop that does
   not depend on the mailbox state.

The planner has no hook. It blocks on `watch --as woebbe` within a turn and checks back
across longer gaps.

## The guard hook

`guard_hook.py` runs before every shell command and file read the executor makes in Cursor.
While a session is armed it denies the actions listed as enforced in `GUARDRAILS.md` and tells
the executor to post BLOCKED instead. It runs with `failClosed`, so if the hook itself fails, the
action is blocked. Every denial is written to `log/guard.log`. The rules are regular expressions
in the script, and `config.json` can extend or replace them under `guard`.

It is a strong layer, not a sandbox. It sees shell commands and file reads. It does not see
edits made through Cursor's file-edit tool, and a command it does not recognize gets through.
Keep Cursor's own auto-run allowlist narrow as well (see the README).

## Transport integrity

Protocol state (`state.json`, the inbox files, `log/`) is written only by `mbx.py`. A side that
cannot run the CLI stops the session rather than writing those files by hand. Hand-written state
skips the log and leaves a gap in the audit trail.

## Discipline

The mailbox removes the copy-paste, not the checking. The executor cites artifacts by path, and
the planner reads those artifacts rather than trusting the summary. Automation makes that step
easier to skip, which is the reason not to skip it.
