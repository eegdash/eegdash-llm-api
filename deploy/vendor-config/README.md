# Vendored eegdash-tagger config

`few_shot_examples.json` and `prompt.md` copied from the local eegdash-llm-tagger
checkout. The tagger wheel/git install does NOT ship these, and the library's default
paths resolve to a non-existent location when installed non-editably. Dockerfile.prod
COPYs these to /app/config and compose sets FEW_SHOT_PATH / PROMPT_PATH to point here.

NOTE: this is the local working copy (may include uncommitted edits) — it defines the
deployed config_hash. To deploy the repo's committed version instead, re-copy from a
clean checkout of eegdash-llm-tagger@main.
