# Team Documentation Workflow

Collaboration centered on the `docs/` directory.

## Knowledge Architecture (PARA)

Based on the PARA methodology. Classify information at the point of capture — no staging area.

```
docs/
├── index.md           # Document index
├── projects/          # Time-bound efforts with clear deliverables
├── areas/             # Ongoing responsibilities requiring maintained standards
├── resources/         # Topics and references with potential future value
└── archives/          # Completed, abandoned, or shelved inactive items
```

## Core Principles (CODE)

Documentation exists to advance work, not to hoard information. Four steps when handling information:

1. **Capture**: Only record what materially advances the project. When in doubt, leave it out.
2. **Organize**: Action-oriented. Resist the urge to organize for organization's sake — structure should be just enough.
3. **Distill**: Refactor over append. When you learn something new or hit a pitfall, integrate it into the document body — don't pile a changelog at the bottom.
4. **Express**: Every document must point to a next step. Split unwieldy documents proactively. Active documents must note the current blocker or next action.

## Collaboration Lifecycle

**Sync**

At the start of each session, you must read `index.md` and load the documents needed for the task at hand.

**Execute**
- Update relevant documents as you go. When a new problem or idea arises, create a document in the appropriate PARA directory.
- Record *why* a decision was made, not just *what* was done.

**Commit**

When a session wraps up:
- Update the TL;DR and status at the top of each modified document.
- Update `index.md` to keep the global routing table current.

## index.md Specification

The document index — nothing more:

| Path | TL;DR |
| --- | --- |
