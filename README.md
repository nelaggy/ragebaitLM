# ragebaitLM

for measuring how angry you get at your coding agents.

coding agent best practice is to not anthropomorphise, but it's really hard to do that when they are trained to induce anthropomorphisation. getting pissed at your agent is a near universal experience, but i see lots of discussion saying different models feel different and i thought it would be really interesting to quantify that.

`ragebaitLM` ingests session logs from **Codex**, **Claude Code**, **OpenCode**,
**Pi**, **Cursor** (IDE composers and `cursor-agent` CLI transcripts), and
**VS Code Copilot Chat**, scores your messages with on sentiment, blames the
model that produced the output just before it and stores statistics about it to
produce a short report.

## Install

Python 3.11+ and pip. The base install uses the lightweight `lexicon` engine;
the default `hybrid` engine needs the optional `ml` extra (torch + transformers).

```bash
python3 -m venv .venv
source .venv/bin/activate

pip install -e .            # lexicon engine only (no torch)
pip install -e ".[ml]"      # required for hybrid / transformer engine
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

| Engine        | Mood quantification                                       |
|---------------|-----------------------------------------------------------|
| `transformer` | `transformer only` |
| `lexicon`     | `metrics only` |
| `hybrid`      | `transformer score + some metrics` |

If the ML dependencies are missing, `hybrid` falls back to `lexicon`
with a warning and `transformer` exits with an error.

## How mood is quantified

Mood is a single signed score in `[-100, 100]`: `+100` is pure joy, `0` is neutral, `-100`
is unbridled rage. It's computed as the difference between positive and negative evidence, clamped to `[-1, 1]` and
then scaled to `[-100, 100]`.

**Transformer-only mood** (no lexicon):

The sentiment model (`cardiffnlp/twitter-roberta-base-sentiment-latest`) returns
probabilities over `negative`, `neutral`, `positive`.
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

Each scored prompt is blamed on the nearest preceding assistant message. A
prompt with no preceding assistant (the opening turn of a session) is stored but
excluded from report statistics, since there is no model response to attribute it
to.

## Storage

`data/ragebaitlm.db` (SQLite) stores session metadata, per-message
`text_sha256`/length (not text), sentiment/mood scores, and attribution. Pass
`--cache-text` to also persist raw text for local experimentation; it is
gitignored.

## Tests

```bash
pytest
```
