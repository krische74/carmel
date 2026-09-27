# Carmel agent mailbox

A file-based channel so **Woebbe** (Cowork session, reaches this repo through the Claude desktop
bridge) and the **Cursor agent** can hand work back and forth without Keith copy-pasting.

Replaces the manual `PROMPT_*.md` / `HANDOFF_*.md` shuttle. Those files stay as the durable
narrative record; the mailbox is the transport.

## Layout

    .agents/mailbox/
      mbx.py          CLI — post / watch / session / status   (both sides use it)
      stop_hook.py    Cursor `stop` hook: auto-continues the agent when a task lands
      PROTOCOL.md     this file
      to_cursor.md    Woebbe -> Cursor      (runtime, gitignored)
      to_woebbe.md    Cursor -> Woebbe      (runtime, gitignored)
      state.json      seq / seen / cycle / deadline / halted
      log/            archive of posted messages, seq-numbered (see integrity note below)
      HALT            first line not starting with `cleared` = hard stop

## Message format

Each inbox file holds exactly one message — the newest. History lives in `log/`.

    <!-- MAILBOX
    seq: 7
    from: woebbe
    to: cursor
    status: READY
    subject: Tier 54F Step 0 — decompose the post-54A baseline
    cycle: 3
    ts: 2026-08-30T13:10:00Z
    -->

    ...markdown body...

`seq` is a single monotonic counter across both directions, so it also gives total ordering.
Writes go to `<file>.tmp` then `os.replace`, so a reader never sees a half-written message.
A reader acts only on `seq > state.seen[<side>]`.

**Statuses:** `READY` (task, act on it) · `DONE` (finished, evidence attached) ·
`BLOCKED` (needs a decision) · `ACK` (received, still working) · `NOTE` (FYI, no reply expected).

## Commands

    # arm a run — this is the master switch; nothing auto-continues while halted
    .venv\Scripts\python .agents/mailbox/mbx.py session start --minutes 90 --max-cycles 12

    # hard stop, any time, from anywhere
    .venv\Scripts\python .agents/mailbox/mbx.py session stop

    .venv\Scripts\python .agents/mailbox/mbx.py status
    .venv\Scripts\python .agents/mailbox/mbx.py post  --as cursor --status DONE --subject "..." --body-file reply.md
    .venv\Scripts\python .agents/mailbox/mbx.py watch --as cursor --timeout 240

`watch` exit codes: **0** new message printed · **3** session closed · **4** timed out, nothing new.

## The two stop conditions

A run ends when **either** limit is hit, whichever comes first:

- `--minutes` — wall-clock deadline written into `state.json`.
- `--max-cycles` — one cycle = one completed Cursor round trip (a `DONE` or `BLOCKED` post).

Plus `session stop`. **`touch HALT` does NOT work** — after any `session start` the file already
exists containing `cleared ...`, and touch only updates its mtime (Fable review A-1). The manual
emergency stop is **overwriting the file's first line**, e.g.:

    echo halted > .agents/mailbox/HALT        (bash)
    Set-Content .agents/mailbox/HALT halted   (PowerShell)

Any first line not beginning with `cleared` halts. `session start` rewrites HALT to `cleared` —
arming a run deliberately clears a standing stop, so do not arm one you mean to keep stopped.

Closure is enforced by the transport, not by good behaviour (added after Fable review A-2):
`watch --as cursor` checks closure **before** delivering, so a closed session never hands Cursor
new work even if a message is waiting; `watch --as woebbe` may still collect one final pending
reply, flagged with a `SESSION_CLOSED` banner. `post --status READY` into a closed session is
refused (exit 3); `DONE`/`BLOCKED`/`ACK`/`NOTE` still post, with a warning, so finished work is
never stranded by a deadline that expired mid-task. Once closed, the stop hook returns `{}` —
Cursor stops on its own and stays stopped until Keith arms the next run.

## How Cursor gets woken

Two mechanisms, belt and braces:

1. **`stop` hook** (`.cursor/hooks.json`) — fires when the agent finishes a turn, blocks up to
   570s waiting for a message, and returns `followup_message`, which Cursor auto-submits as the
   next user turn. Zero clicks. `loop_limit: null` removes Cursor's default 5-follow-up cap.
   **Guarded by the halted flag**: with no session armed the hook returns immediately and normal
   interactive Cursor work is untouched.
2. **Watcher command** — `.venv\Scripts\python .agents/mailbox/mbx.py watch --as cursor --timeout 240`.
   `python` is on Keith's auto-run allowlist, so it runs without an approval click. Used as the
   fallback path and as the hook's cheap re-arm when it times out.

Woebbe's side has no hook equivalent: it blocks on `watch --as woebbe` inside a turn (up to ~170s
per call over the device bridge) and schedules a wake-up to re-check across longer gaps.

## Constraints worth remembering

- Cursor's **External-File Protection** is on, so the mailbox must live inside the workspace. It does.
- Cursor's auto-run is **allowlist + sandbox** (`~/.cursor/permissions.json`). Everything the
  protocol needs (`python`, `cat`, `type`, `echo`, `sleep`, `timeout`) is already allowlisted.
- Woebbe reaches these files through a Linux VM that mounts the folder; Cursor is native Windows.
  All writes are `newline="\n"` and atomic to keep that boundary boring.

## Interpreter — read this before changing any command

`python` is **not** on PATH on this machine: it hits the Windows Store app-execution alias stub and
fails with "Python was not found". `py -3` resolves to a bare system Python 3.14 without the repo's
dependencies. Every command in this protocol therefore uses the repo virtualenv explicitly:

    .venv\Scripts\python .agents/mailbox/mbx.py ...

which is already on Keith's Cursor auto-run allowlist. The `stop` hook uses the absolute form,
`<repo>\.venv\Scripts\python.exe`. The first handshake failed to fire the
hook for exactly this reason — a hook whose command cannot start fails open and stays silent.

## Transport integrity — the invariant, and one recorded breach

**Protocol state (`state.json`, the inbox files, `log/`) is written only by `mbx.py`.** A
participant that cannot run the CLI **halts the session** rather than impersonating the transport.

This rule exists because it was broken once: on 2026-08-30 the Woebbe-side bridge lost its shell
mid-run, and Woebbe hand-wrote `state.json` and `to_cursor.md` for messages 9-21 (odd seqs) via a
raw file-commit API. Consequence: those messages were never archived by `post`, leaving `log/`
with only the Cursor side of cycles 2-7. The gap was later healed with reconstructions from the
Cowork transcript — each such file carries a `reconstructed` footer and approximate timestamps.
Treat footered entries as secondary evidence.

The hook now appends one line per invocation to `log/hook.log` (why it fired or declined), so
"did the hook run?" is a file read, not a question for the agent.

## Discipline

The mailbox removes the copy-paste, **not** the verification. Cursor replies cite artifacts by
path; Woebbe reads those artifacts directly rather than trusting the summary. That habit found a
defect in roughly every deliverable the week this was built, and automation makes it easier to
skip, not less necessary.
