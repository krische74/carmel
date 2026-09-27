#!/usr/bin/env python
"""Carmel agent mailbox — file-based handoff between Woebbe (Cowork) and Cursor.

Subcommands
-----------
  session start --minutes N [--max-cycles N]   arm a run
  session stop                                 halt immediately
  status                                       print state as JSON
  post --as {woebbe,cursor} --status {READY,DONE,BLOCKED,ACK,NOTE} [--subject S] [--body-file F | --stdin]
  watch --as {woebbe,cursor} [--timeout S] [--poll S]

watch exit codes:  0 = new message printed | 3 = session closed | 4 = timed out, nothing new
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
STATE = os.path.join(HERE, "state.json")
LOGDIR = os.path.join(HERE, "log")
HALTFILE = os.path.join(HERE, "HALT")
SIDES = ("woebbe", "cursor")
OTHER = {"woebbe": "cursor", "cursor": "woebbe"}

DEFAULT_STATE = {
    "seq": 0,
    "seen": {"woebbe": 0, "cursor": 0},
    "cycle": 0,
    "max_cycles": 0,
    "deadline_utc": None,
    "halted": True,
    "session_started_utc": None,
    "last_reason": "never started",
}


def utcnow() -> _dt.datetime:
    return _dt.datetime.now(_dt.UTC)


def iso(ts: _dt.datetime) -> str:
    return ts.strftime("%Y-%m-%dT%H:%M:%SZ")


def inbox(side: str) -> str:
    return os.path.join(HERE, f"to_{side}.md")


# ---------------------------------------------------------------- state io
# No lock file: the Woebbe side reaches this folder through a bridge that cannot delete
# files, so a lock could never be released. Instead every write re-reads the file and
# merges monotonic fields (seq, cycle, seen) with max(), which is safe for a two-party,
# human-paced channel.
def read_state() -> dict:
    if not os.path.exists(STATE):
        return dict(DEFAULT_STATE, seen=dict(DEFAULT_STATE["seen"]))
    for _ in range(5):
        try:
            with open(STATE, encoding="utf-8") as fh:
                s = json.load(fh)
            break
        except (json.JSONDecodeError, OSError):
            time.sleep(0.1)
    else:
        return dict(DEFAULT_STATE, seen=dict(DEFAULT_STATE["seen"]))
    out = dict(DEFAULT_STATE, seen=dict(DEFAULT_STATE["seen"]))
    out.update(s)
    out["seen"] = dict(DEFAULT_STATE["seen"], **(s.get("seen") or {}))
    return out


def write_state(state: dict) -> None:
    tmp = STATE + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(state, fh, indent=2, sort_keys=True)
    os.replace(tmp, STATE)


def log_max_seq() -> int:
    """Highest seq present in log/ — lets seq self-heal if state.json is lost."""
    best = 0
    try:
        for name in os.listdir(LOGDIR):
            head = name.split("_", 1)[0]
            if head.isdigit():
                best = max(best, int(head))
    except OSError:
        pass
    return best


def _merge(mine: dict, disk: dict) -> dict:
    out = dict(disk)
    out.update(mine)
    # halted is sticky-true across a merge: a concurrent `session stop` must never be
    # undone by a writer holding a stale halted=False (Fable review A-3).
    out["halted"] = bool(mine.get("halted", True)) or bool(disk.get("halted", True))
    out["seq"] = max(int(mine.get("seq", 0)), int(disk.get("seq", 0)))
    out["cycle"] = max(int(mine.get("cycle", 0)), int(disk.get("cycle", 0)))
    out["seen"] = {
        side: max(
            int((mine.get("seen") or {}).get(side, 0)),
            int((disk.get("seen") or {}).get(side, 0)),
        )
        for side in SIDES
    }
    return out


def update_state(fn, monotonic: bool = True):
    s = read_state()
    result = fn(s)
    write_state(_merge(s, read_state()) if monotonic else s)
    return result


# ------------------------------------------------------------ session gate
def halt_active() -> bool:
    """HALT is active when the file exists and has not been retired with a `cleared` marker."""
    if not os.path.exists(HALTFILE):
        return False
    try:
        with open(HALTFILE, encoding="utf-8") as fh:
            return not fh.readline().strip().lower().startswith("cleared")
    except OSError:
        return True


def closed_reason(state: dict) -> str | None:
    """Return a human reason if the session is closed, else None."""
    if halt_active():
        return "HALT file present"
    if state.get("halted", True):
        return "session not started (halted)"
    mc = int(state.get("max_cycles") or 0)
    if mc and int(state.get("cycle", 0)) >= mc:
        return f"cycle cap reached ({state['cycle']}/{mc})"
    dl = state.get("deadline_utc")
    if dl:
        try:
            deadline = _dt.datetime.strptime(dl, "%Y-%m-%dT%H:%M:%SZ").replace(
                tzinfo=_dt.UTC
            )
        except ValueError:
            return None
        if utcnow() >= deadline:
            return f"time budget expired at {dl}"
    return None


# ---------------------------------------------------------------- messages
HDR_OPEN = "<!-- MAILBOX"
HDR_CLOSE = "-->"


def parse(path: str) -> dict | None:
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as fh:
            text = fh.read()
    except OSError:
        return None
    if not text.startswith(HDR_OPEN):
        return None
    end = text.find(HDR_CLOSE)
    if end == -1:
        return None
    hdr = {}
    for line in text[len(HDR_OPEN):end].splitlines():
        if ":" in line:
            k, _, v = line.partition(":")
            hdr[k.strip()] = v.strip()
    hdr["body"] = text[end + len(HDR_CLOSE):].lstrip("\n")
    hdr["raw"] = text
    try:
        hdr["seq"] = int(hdr.get("seq", 0))
    except ValueError:
        hdr["seq"] = 0
    return hdr


def cmd_post(args) -> int:
    if args.stdin:
        body = sys.stdin.read()
    elif args.body_file:
        with open(args.body_file, encoding="utf-8") as fh:
            body = fh.read()
    else:
        print("post: need --stdin or --body-file", file=sys.stderr)
        return 2

    sender = args.as_side
    target = OTHER[sender]

    reason = closed_reason(read_state())
    if reason and args.status == "READY":
        # New tasks cannot enter a closed session (Fable review A-2). DONE/BLOCKED/ACK/NOTE
        # still post so completed work is never stranded by a deadline expiring mid-run.
        print(f"POST_REFUSED (READY into closed session): {reason}")
        return 3
    if reason:
        print(f"WARNING: session closed ({reason}); posting {args.status} anyway so the reply is not lost")

    def bump(s):
        s["seq"] = max(int(s.get("seq", 0)), log_max_seq()) + 1
        if sender == "cursor" and args.status in ("DONE", "BLOCKED"):
            s["cycle"] = int(s.get("cycle", 0)) + 1
        return (s["seq"], s["cycle"], s.get("max_cycles", 0))

    seq, cycle, max_cycles = update_state(bump)
    ts = iso(utcnow())
    header = (
        f"{HDR_OPEN}\n"
        f"seq: {seq}\n"
        f"from: {sender}\n"
        f"to: {target}\n"
        f"status: {args.status}\n"
        f"subject: {args.subject or '(none)'}\n"
        f"cycle: {cycle}\n"
        f"ts: {ts}\n"
        f"{HDR_CLOSE}\n\n"
    )
    text = header + body.rstrip("\n") + "\n"

    dest = inbox(target)
    tmp = dest + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
    os.replace(tmp, dest)

    os.makedirs(LOGDIR, exist_ok=True)
    stamp = ts.replace(":", "").replace("-", "")
    with open(
        os.path.join(LOGDIR, f"{seq:04d}_{sender}_to_{target}_{stamp}.md"),
        "w",
        encoding="utf-8",
        newline="\n",
    ) as fh:
        fh.write(text)

    print(f"POSTED seq={seq} {sender}->{target} status={args.status} cycle={cycle}/{max_cycles or '-'}")
    return 0


def cmd_watch(args) -> int:
    side = args.as_side
    path = inbox(side)
    deadline = time.time() + args.timeout
    while True:
        state = read_state()
        reason = closed_reason(state)
        msg = parse(path)
        seen = int(state.get("seen", {}).get(side, 0))
        pending = bool(msg and msg["seq"] > seen)
        # Closed-session rule (Fable review A-2): closure is checked BEFORE delivery, so
        # the transport refuses to hand new work to cursor once the session is closed.
        # woebbe may still collect one final pending reply (with a banner), so a DONE that
        # lands exactly on the cycle cap is not stranded; the next watch returns 3.
        if reason and (side == "cursor" or not pending):
            print(f"SESSION_CLOSED: {reason}")
            return 3
        if pending:
            def mark(s, _seq=msg["seq"]):
                s.setdefault("seen", {})[side] = _seq
                return None
            update_state(mark)
            if reason:
                print(f"SESSION_CLOSED: {reason} (final pending message delivered below)")
            print(f"NEW_MESSAGE seq={msg['seq']} from={msg.get('from')} status={msg.get('status')}")
            print("=" * 72)
            print(msg["raw"].rstrip("\n"))
            print("=" * 72)
            return 0
        if time.time() >= deadline:
            print(f"NO_NEW_MESSAGE (waited {args.timeout}s, last seen seq={seen})")
            return 4
        time.sleep(args.poll)


def cmd_session(args) -> int:
    if args.action == "start":
        with open(HALTFILE, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(f"cleared {iso(utcnow())}\n")
        end = utcnow() + _dt.timedelta(minutes=args.minutes)

        def start(s):
            s["halted"] = False
            s["cycle"] = 0
            s["max_cycles"] = args.max_cycles
            s["deadline_utc"] = iso(end)
            s["session_started_utc"] = iso(utcnow())
            s["last_reason"] = "running"
            return None

        update_state(start, monotonic=False)
        print(f"SESSION STARTED until {iso(end)} ({args.minutes} min), max_cycles={args.max_cycles or 'unlimited'}")
        return 0

    if args.action == "stop":
        with open(HALTFILE, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(f"halted {iso(utcnow())}\n")

        def stop(s):
            s["halted"] = True
            s["last_reason"] = "stopped by operator"
            return None

        update_state(stop)
        print("SESSION HALTED")
        return 0
    return 2


def cmd_status(args) -> int:
    s = read_state()
    s["closed_reason"] = closed_reason(s)
    s["now_utc"] = iso(utcnow())
    for side in SIDES:
        m = parse(inbox(side))
        s[f"inbox_{side}"] = (
            {"seq": m["seq"], "from": m.get("from"), "status": m.get("status"), "subject": m.get("subject")}
            if m else None
        )
    print(json.dumps(s, indent=2, sort_keys=True))
    return 0


def main() -> int:
    p = argparse.ArgumentParser(prog="mbx")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("post")
    sp.add_argument("--as", dest="as_side", choices=SIDES, required=True)
    sp.add_argument("--status", default="READY",
                    choices=["READY", "DONE", "BLOCKED", "ACK", "NOTE"])
    sp.add_argument("--subject", default="")
    sp.add_argument("--body-file")
    sp.add_argument("--stdin", action="store_true")
    sp.set_defaults(func=cmd_post)

    sw = sub.add_parser("watch")
    sw.add_argument("--as", dest="as_side", choices=SIDES, required=True)
    sw.add_argument("--timeout", type=float, default=240.0)
    sw.add_argument("--poll", type=float, default=2.0)
    sw.set_defaults(func=cmd_watch)

    ss = sub.add_parser("session")
    ss.add_argument("action", choices=["start", "stop"])
    ss.add_argument("--minutes", type=int, default=60)
    ss.add_argument("--max-cycles", type=int, default=0)
    ss.set_defaults(func=cmd_session)

    st = sub.add_parser("status")
    st.set_defaults(func=cmd_status)

    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
