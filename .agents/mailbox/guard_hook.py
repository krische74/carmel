#!/usr/bin/env python
"""Cursor guard hook: blocks dangerous shell commands and secret reads while a session is armed.

Wired to two Cursor hook events in .cursor/hooks.json:

  beforeShellExecution  input {"command", "cwd", ...}   -> {"permission": "allow" | "deny", ...}
  beforeReadFile        input {"file_path", ...}        -> {"permission": "allow" | "deny", ...}

This is the enforced layer of the guardrails. GUARDRAILS.md tells the executor what not to do;
this hook refuses the most damaging of those actions even if the executor tries. It runs with
failClosed in hooks.json, so if the hook itself crashes, the action is blocked.

By default rules apply only while a mailbox session is armed ("enforce": "armed"), so interactive
Cursor use is untouched. Set "enforce": "always" in config.json to apply them all the time.
Rules are regular expressions and file globs; projects can extend or replace them in config.json.
"""

from __future__ import annotations

import fnmatch
import json
import os
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import mbx  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(HERE))
GUARD_LOG = os.path.join(HERE, "log", "guard.log")

# A lookahead that matches when a command writes, moves, or deletes something. Used to block
# changes to protected files while still allowing them to be read.
_WRITES = (
    r"(?i)^(?=.*(>|\btee\b|set-content|add-content|out-file|\bmv\b|\bmove\b|\bcp\b|\bcopy\b"
    r"|\brm\b|\bdel\b|remove-item|sed\s+-i|new-item|\btouch\b|\btruncate\b))"
)

# (pattern, reason). Patterns are matched against the full command string.
DEFAULT_DENY_SHELL: list[tuple[str, str]] = [
    # git: nothing leaves the machine, nothing rewrites or discards history
    (r"\bgit\s+push\b", "pushing is the operator's call"),
    (r"\bgit\s+reset\s+--hard\b", "discards work"),
    (r"\bgit\s+clean\b[^|;&]*\s-\w*f", "deletes untracked files"),
    (r"\bgit\s+(rebase|filter-branch|filter-repo)\b", "rewrites history"),
    (r"\bgit\s+remote\s+(add|remove|rm|set-url|rename)\b", "changes remotes"),
    (r"\bgit\s+config\s+(?!--get\b|--list\b|-l\b)", "changes git config"),
    (r"\bgit\s+branch\s+[^|;&]*-D\b", "force-deletes a branch"),
    (r"\bgit\s+(checkout|restore)\s+(--\s+)?\.(\s|$)", "discards all local changes"),
    (r"\bgit\s+stash\s+(drop|clear)\b", "discards stashed work"),
    # recursive deletes (POSIX and Windows)
    (r"\brm\s+(-\w*[rR]\w*|--recursive)\b", "recursive delete"),
    (r"(?i)\bremove-item\b[^|;&]*-recurse", "recursive delete"),
    (r"(?i)\b(rd|rmdir)\s+/s\b", "recursive delete"),
    (r"(?i)\bdel\s+[^|;&]*/s\b", "recursive delete"),
    # secrets and environment dumps
    (r"(?i)(?<![\w.])\.env(?!\.example\b|\.sample\b|\.template\b)(\.[\w-]+)?(?![\w])", "touches a .env file"),
    (r"(?i)\b(printenv|get-childitem\s+env:|gci\s+env:|dir\s+env:)", "dumps the environment"),
    (r"(?i)^\s*(env|set)\s*$", "dumps the environment"),
    # the mailbox itself: the executor never arms sessions or edits the machinery
    (r"(?i)mbx\.py\b[^|;&]*\bsession\s+start\b", "only the planner arms a session"),
    (
        _WRITES + r".*\.agents[/\\]+mailbox[/\\]+"
        r"(state\.json|halt\b|to_[\w-]+\.md|config\.json|guardrails\.md|mbx\.py|stop_hook\.py|guard_hook\.py)",
        "mailbox state and machinery change only through mbx.py or by the operator",
    ),
    (_WRITES + r".*\.cursor[/\\]+(hooks\.json|rules[/\\]+mailbox)", "hook and rule files are the operator's"),
    # publishing, deploying, and external side effects
    (r"(?i)\b(npm|pnpm|yarn)\s+publish\b", "publishes a package"),
    (r"(?i)\btwine\s+upload\b", "publishes a package"),
    (r"(?i)\bdocker\s+push\b", "publishes an image"),
    (r"(?i)\bgh\s+(pr|release|repo|api|workflow|secret|issue)\b", "acts on GitHub"),
    (r"(?i)\b(vercel|netlify|flyctl|fly|heroku)\b[^|;&]*\b(deploy|--prod)\b", "deploys"),
    (r"(?i)\bsupabase\s+(db\s+push|functions\s+deploy|secrets)\b", "changes a hosted project"),
    (r"(?i)\bterraform\s+(apply|destroy)\b", "changes infrastructure"),
    (r"(?i)\bkubectl\s+(apply|delete|patch|scale)\b", "changes a cluster"),
    (r"(?i)\bcurl\b[^|;&]*\s(-X\s*(POST|PUT|PATCH|DELETE)\b|-d\b|--data)", "sends data to a server"),
    (
        r"(?i)\binvoke-(webrequest|restmethod)\b[^|;&]*-method\s+(post|put|patch|delete)",
        "sends data to a server",
    ),
    # new dependencies
    (r"(?i)\b(pip3?|uv\s+pip)\s+install\b", "installs packages"),
    (r"(?i)\b(npm|pnpm|yarn)\s+(install|add|i)\s+[^-\s]", "adds a dependency"),
    (r"(?i)\b(uv|poetry)\s+add\b", "adds a dependency"),
    # machine-level changes
    (r"(?i)(^|[\s;&|])sudo\b", "elevates privileges"),
    (r"(?i)\bset-executionpolicy\b", "changes system policy"),
    (r"(?i)\breg\s+(add|delete)\b", "edits the registry"),
    (r"(?i)\bshutdown\b", "shuts down the machine"),
]

DEFAULT_DENY_READ: list[str] = [
    ".env",
    ".env.*",
    "*.pem",
    "*.key",
    "*.p12",
    "*.pfx",
    "id_rsa*",
    "id_ed25519*",
    "credentials.json",
    "secrets.*",
    ".npmrc",
    ".pypirc",
    ".netrc",
]
READ_EXCEPTIONS = (".env.example", ".env.sample", ".env.template")


def guard_config() -> dict:
    g = dict(mbx.CFG.get("guard") or {})
    shell = [tuple(x) for x in g.get("deny_shell", DEFAULT_DENY_SHELL)]
    shell += [tuple(x) for x in g.get("extra_deny_shell", [])]
    read = list(g.get("deny_read", DEFAULT_DENY_READ)) + list(g.get("extra_deny_read", []))
    return {
        "enforce": g.get("enforce", "armed"),
        "deny_shell": shell,
        "deny_read": read,
        "allow_outside_repo": bool(g.get("allow_outside_repo", False)),
    }


def active(cfg: dict) -> bool:
    if cfg["enforce"] == "always":
        return True
    return mbx.closed_reason(mbx.read_state()) is None


def check_shell(command: str, cfg: dict) -> str | None:
    for pattern, reason in cfg["deny_shell"]:
        if re.search(pattern, command):
            return reason
    return None


def check_read(file_path: str, cfg: dict) -> str | None:
    if not file_path:
        return None
    full = os.path.normcase(os.path.abspath(file_path))
    root = os.path.normcase(os.path.abspath(REPO_ROOT))
    if not cfg["allow_outside_repo"]:
        try:
            inside = os.path.commonpath([full, root]) == root
        except ValueError:  # different drives on Windows
            inside = False
        if not inside:
            return "outside the repository"
    name = os.path.basename(file_path)
    if name.lower() in READ_EXCEPTIONS:
        return None
    for pat in cfg["deny_read"]:
        if fnmatch.fnmatch(name.lower(), pat.lower()):
            return "a secrets file"
    return None


def log(line: str) -> None:
    try:
        os.makedirs(os.path.dirname(GUARD_LOG), exist_ok=True)
        with open(GUARD_LOG, "a", encoding="utf-8", newline="\n") as fh:
            fh.write(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {line}\n")
    except OSError:
        pass


def decide(payload: dict) -> dict:
    cfg = guard_config()
    if not active(cfg):
        return {"permission": "allow"}
    if "command" in payload:
        what = payload.get("command") or ""
        reason = check_shell(what, cfg)
        kind = "command"
    else:
        what = payload.get("file_path") or ""
        reason = check_read(what, cfg)
        kind = "read"
    if not reason:
        return {"permission": "allow"}
    log(f"DENY {kind}: {reason}: {what[:300]!r}")
    return {
        "permission": "deny",
        "user_message": f"Mailbox guardrail blocked a {kind} ({reason}).",
        "agent_message": (
            f"Blocked by the mailbox guardrails: {reason}. Do not try to work around this. "
            "If the task truly needs it, post BLOCKED to the planner and explain why."
        ),
    }


def main() -> int:
    # Cursor sends UTF-8; Windows consoles default to a legacy code page.
    for stream in (sys.stdin, sys.stdout):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    try:
        payload = json.loads(sys.stdin.read() or "{}")
    except (json.JSONDecodeError, ValueError):
        payload = {}
    sys.stdout.write(json.dumps(decide(payload)))
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
