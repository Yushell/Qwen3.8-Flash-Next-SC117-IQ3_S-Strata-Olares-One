# Patches

## `patch_strata_sanitize.py`: quoted-XML tool tags become text

**What:** a serve-layer patch to Strata's `serve/server.py`. Quoted XML tool tags in
**client-supplied** strings (`<tool_call>`, `<arg_key>`, `<arg_value>`, `<function>`,
`<parameter>`, and the `<|name|>` spelling) are rewritten to `[name]` before the prompt is
rendered, while the framing the chat template itself adds is left untouched. It runs as
`_sanitize(messages, tools)` before `prepare()`.

**Why:** without it, a client can put chat-template framing into a message and make the model
treat text as a real tool call or a real turn boundary (the 2026-09-30 special-token smuggling
finding). Upstream later rendered control-token text as text (its tests pin that), which retired
the pipe-token half; the quoted XML half is not covered upstream, so the patch stays.

**Apply:** `python3 patch_strata_sanitize.py --check` then without `--check`. Re-apply after
**every** engine pull: `git pull` / `./update.sh` overwrites `serve/server.py`. The patcher
refuses when its anchor drifted; check it against the incoming file before pulling when a release
touches `serve/`.

**Tests:** from the Strata checkout, `PYTHONPATH=. python3 serve/test_sanitize.py` and
`serve/test_strata_quoted_xml.py`, plus the full suite
(`.venv/bin/python -m unittest discover -s serve -p "test_*.py" -q`).

Apache-2.0, like the rest of this repo. It modifies Strata (MIT); Strata's LICENSE applies to the
code it patches.
