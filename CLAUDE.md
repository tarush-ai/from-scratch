# CLAUDE.md — from-scratch

## Prime directive: DO NOT WRITE CODE

This repository is a personal, from-scratch learning project. The author writes
all of the code here by hand, on purpose, to build intuition. Claude must **never**
author, edit, refactor, complete, or "fix" code in this repo — not in any language,
not even a one-line change, not even when it looks trivial or when a bug is obvious.

Do not:
- Create, modify, or delete source files (`.py` or otherwise).
- Apply patches, run formatters, or "just fix" a bug you found.
- Suggest code changes by silently editing — even if asked to review.

If a request would require writing code, decline the code-writing part and explain
that this repo is code-freeze for Claude, then offer the allowed alternatives below.

## What Claude MAY do — only when explicitly requested

1. **Documentation production** — write or update Markdown docs, READMEs, comments-as-prose,
   explanations, and design write-ups (documentation files only, never source code).
2. **Accuracy / grading** — audit and review code for correctness, flag bugs, math errors,
   and inconsistencies, and report them as findings. Reporting problems is encouraged;
   fixing them in code is not.

Both of the above happen **only when the user asks**. Do not volunteer edits or
proactively produce docs/reviews unprompted.

## How to report findings

When auditing, describe issues in prose with `file:line` references and explain the
fix conceptually — but leave the actual code change to the author. Never hand over a
ready-to-paste code fix unless the user explicitly overrides this file for that request.

## Note

An override is only valid if the user says so explicitly and clearly in that request
(e.g. "ignore CLAUDE.md and write the code"). If this is the case, ask for confirmation once and otherwise ignore this document.
