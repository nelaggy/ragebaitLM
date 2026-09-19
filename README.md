# ragebaitLM

Analyse coding-agent sessions for user mood (joy ↔ rage).

`ragebaitLM` ingests session logs from **Codex**, **Claude Code**, **OpenCode**, and
**Pi**, scores *your* messages with a sentiment model, attributes each message to
the model that produced the output just before it, stores structured stats
(never the full chats), and renders a self-contained HTML report.

## Install

Python 3.11+ and pip. The base install uses the lightweight `lexicon` engine;
the default `hybrid` engine needs the optional `ml` extra (torch + transformers).

```bash
python3 -m venv .venv
source .venv/bin/activate

pip install -e .            # lexicon engine only (no torch)
pip install -e ".[ml]"      # hybrid / transformer engine
pip install -e ".[ml,dev]"  # + test dependencies
```

## Usage

```bash
# 1. Import + score logs into data/ragebaitlm.db
ragebaitlm sync                       # all harnesses
ragebaitlm sync --harness opencode    # one harness
ragebaitlm sync --since 2026-01-01    # time filter

# 2. Print summary / model mood
ragebaitlm analyze

# 3. Build a self-contained HTML report
ragebaitlm report --out data/report.html

# Inspect what was discovered without writing anything
ragebaitlm list-harnesses
```

After `pip install`, the `ragebaitlm` console script is available. You can also run
it as a module without installing the script: `python -m ragebaitlm --help`.

### Sentiment engines

| Engine        | Sentiment model                              | Mood quantification                                       |
|---------------|----------------------------------------------|-----------------------------------------------------------|
| `hybrid`      | `cardiffnlp/twitter-roberta-base-sentiment-latest` | transformer polarity + mood lexicon (default)       |
| `transformer` | same                                          | transformer probabilities only                            |
| `lexicon`     | none (VADER + custom lexicon)                 | lexicon only (no torch)                                   |

If the ML dependencies are missing, `hybrid` falls back to the lexicon engine
with a warning and `transformer` exits with an error.

## How mood is quantified

Mood is a single signed score in `[-1, 1]`: **`+1` is joy, `0` is neutral, `-1`
is rage**. It is the *difference* between positive and negative evidence, not a
one-sided intensity.

The sentiment model (`cardiffnlp/twitter-roberta-base-sentiment-latest`) returns
probabilities over `negative`, `neutral`, `positive`.

**Transformer-only mood** (no lexicon):

```
mood = clamp(positive - negative, -1, 1)
```

**Lexicon mood** contrasts positive evidence (VADER positivity plus praise
markers such as "thanks", "perfect", "works") with negative evidence (VADER
negativity plus profanity, ALL-CAPS ratio, `!` runs, and correction/urgency
markers such as "no", "stop", "again", "still", "I said", "why", "broken"):

```
positive_strength = clamp(0.60 * vader_pos + 0.40 * joy_markers, 0, 1)
negative_strength = clamp(0.40 * vader_neg + 0.18 * profanity + 0.14 * caps
                          + 0.10 * exclaim + 0.10 * markers + 0.08 * negative_terms, 0, 1)
lexicon_mood      = positive_strength - negative_strength
```

**Hybrid mood** blends the two:

```
mood = clamp(0.65 * transformer_mood + 0.35 * lexicon_mood, -1, 1)
```

Every fired signal is recorded in `message_stat.mood_signals_json`.

## Storage

`data/ragebaitlm.db` (SQLite) stores session metadata, per-message
`text_sha256`/length (not text), sentiment/mood scores, attribution, and
revision signals. Pass `--cache-text` to also persist raw text for local
experimentation; it is gitignored.

The database schema is versioned. When it changes, `ragebaitlm sync` rebuilds the
derived tables automatically; no migration is needed.

## Subagent sessions

Harnesses create child sessions for delegated/subagent work (OpenCode `task`
sessions, Claude Code `subagents/*.jsonl`, Codex `agent_path` rollouts). These
contain one synthetic prompt and no human turns, so they are stored for
provenance but **excluded from turn statistics**. The report header shows how
many were excluded.

The remaining "one-turn" sessions are genuine: a single directive followed by a
long autonomous run. On a real OpenCode corpus the median single-turn session
lasted ~3 minutes and produced ~2.8k output tokens.

## Revert / undo / edit

Revision signals are captured on ingest into `revision_signal`:

- OpenCode: `session.revert` points and `message.removed` events
- Pi: branch/`branch_summary` leaves (tree `parentId` divergence)
- Codex: `:codex-annotation` feedback markers
- Claude Code: interruption / sidechain markers where present

Dedicated visualisations for these are a follow-up; counts are shown in the report.

## Tests

```bash
pytest
```
