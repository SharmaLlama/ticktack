## Summary

<!-- Link to the written task or issue and state the single purpose of this PR in 1-3 sentences. -->
- **Task / Issue:**
- **Purpose:**

## Scope & Shape Check

- [ ] Single purpose, linked to a written task or issue.
- [ ] Changed `src/ticktack/` code lines $\le 400$ (tests, notebooks, and docs excluded).
- [ ] New or edited functions are $\le 80$ AST code lines and $\le 8$ parameters excluding `self`/`cls` (or carry a Carbon-approved `# shape-exempt(carbon): <specific >=15-char technical reason>` comment; no contortions to dodge the limit).
- [ ] Checked for existing helpers before adding new ones; reused helpers listed below.
- [ ] New public names (if any) have a docs example, a test, and an updated `tests/public_surface_snapshot.json`.

**Reused existing helpers / functions:**
-

**Line-budget justification** *(required by CI only if total `src/ticktack/` code lines exceed `.github/ci_ratchets.json`; leave blank otherwise)*:
Line-budget-reason:

## Evidence Table

<!-- Every factual, numerical, or behavioural claim in this PR must map to a test name or a verbatim command + output. Delete any claim without a row. -->

| Claim | Verification (test name or command + output) |
|---|---|
| | |

## Review Gates

- [ ] **Gate 1 (Scope):** Single purpose, within size limit, linked to a written task.
- [ ] **Gate 2 (CI):** `ruff`, complexity & line-budget ratchets, coverage floor, public-surface snapshot, fast notebooks, and overlapping-PR trial merge are green.
- [ ] **Gate 3 (Evidence):** Every claim above has a row in the Evidence Table.
- [ ] **Gate 4 (Independent Carbon review):** Reviewed by an independent Carbon instance given only the diff and a 5-line intent.
- [ ] **Gate 5 (Claude review, if applicable):** Required if this PR touches `cbm.py`, `fields.py`, `fisher.py`, `likelihood.py`, `linear.py`, or `solvers.py`, or changes $> 200$ `src/` code lines. Add the `needs-claude-review` label and stop for Ben to start the Claude review.
- [ ] **Gate 6 (Merge):** Ben merges (agents never merge).

---
🤖 **Authors:** @benjaminpope & [`carbon-reviewer[bot]`](https://github.com/apps/carbon-reviewer) *(Gemini 4 Carbon via Antigravity)*
