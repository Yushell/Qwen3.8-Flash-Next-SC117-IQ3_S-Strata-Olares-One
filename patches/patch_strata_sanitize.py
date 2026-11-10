"""Apply or verify Strata's client-framing sanitizer (v4, for Strata v0.1.40).

Usage: python3 patch_strata_sanitize.py [path/to/serve/server.py] [--check]
Backs up each changed file; refuses to replace an unrecognized sanitizer.
Place test_sanitize.py alongside this script to install the regression tests too.

v4 (2026-10-06, Strata v0.1.40 / 82f46a8c): upstream #931 now renders and encodes
control-token text (<|im_start|>, <|im_end|>, <|endoftext|>, <|image_pad|>, ...) as
text and its tests pin it verbatim, so the control-token half of this sanitizer is
retired; the method scrubs only the quoted XML framing tags (tool_call, arg_key,
arg_value, function, parameter) and their <|name|> spelling.  Anchors updated to
0.1.40: the method lands before encode_prompt() and the call goes before the
mark_think_literals() call (which gained `self.literals`).  Re-anchored 2026-10-17
for 0.1.45: encode_prompt() gained `store: bool = True` (token counts), so the
signature line now comes from ENCODE_ANCHORS instead of a single constant; the
method body and the mark_think_literals() call are unchanged, and
mark_think_literals(messages, ...) is still unique (the second call site passes
`msgs`, not `messages` -- a looser anchor would land the patch in the wrong
method).  The v1/v2 in-place
upgrade machinery is retired - rollback to those eras uses the archived patchers
(patch_strata_sanitize.py.v2-0.1.39 / .v3-0.1.39) with that era's checkout.
"""
import argparse
import ast
import hashlib
import os
import shutil
from datetime import datetime
from pathlib import Path

METHOD = r'''    def _sanitize(self, messages, tools):
        """Client text must not smuggle template-level framing into the prompt.

        Control-token text (<|im_start|>, <|im_end|>, <|endoftext|>, <|image_pad|>, ...)
        is upstream's own since 0.1.40: mark_think_literals()/#931 renders and encodes it
        as text, and serve/test_server.py pins that verbatim.  This method covers what
        the frontend does not: quoted <tool_call>, <arg_key>, <arg_value>, <function> and
        <parameter> tags (and their <|name|> spelling) become inert bracketed text in
        every client-supplied string - message fields, reasoning, tool-call names and
        nested arguments, and tool schemas.  Only what the client sent is rewritten; the
        framing the template itself adds is untouched, and image sources are opaque
        transport data, not text rendered by Jinja.
        """
        special = re.compile("<[|](tool_call|arg_key|arg_value|function|parameter)[|]>")
        quoted_xml = re.compile(r'<(/?(?:tool_call|arg_key|arg_value|function|parameter)(?:=[^<>\r\n]*)?)>')

        def scrub(value):
            if isinstance(value, str):
                text = special.sub(lambda t: "<" + t.group(1) + ">", value)
                return quoted_xml.sub(lambda t: "[" + t.group(1) + "]", text)
            if isinstance(value, list):
                return [scrub(item) for item in value]
            if isinstance(value, dict):
                return {scrub(key): scrub(item) for key, item in value.items()}
            return value

        cleaned = []
        for m in messages:
            # Image sources are opaque transport data, not text rendered by Jinja.
            # Only text parts are scrubbed; URLs, paths and base64 stay byte-identical.
            content = m.get("content")
            fields = {key: value for key, value in m.items() if key != "content"}
            m = scrub(fields)
            if isinstance(content, list):
                m["content"] = [scrub(part) if isinstance(part, dict) and part.get("type") in
                                ("text", "input_text", None) else part for part in content]
            else:
                m["content"] = scrub(content)
            cleaned.append(m)
        return cleaned, scrub(tools)
'''
NEW_CALL = '        messages, tools = self._sanitize(messages, tools)\n'
# The encode_prompt signature line is the insertion point for METHOD, and one of these must
# match EXACTLY.  0.1.45 added `store: bool = True` (the token-count path); the older form
# still covers 0.1.40 - 0.1.44.1.  A new signature upstream means a new entry here (read the
# incoming file, never loosen these to a substring).
ENCODE_ANCHORS = (
    '    def encode_prompt(self, messages, tools, kwargs) -> list[int]:\n',
    '    def encode_prompt(self, messages, tools, kwargs, store: bool = True) -> list[int]:\n',
)
MARK = '        marked, marked_tools, changed = mark_think_literals(messages, tools, self.literals)\n'


def encode_anchor(src):
    """The one encode_prompt signature line this source carries, or None."""
    hits = [anchor for anchor in ENCODE_ANCHORS if src.count(anchor) == 1]
    return hits[0] if len(hits) == 1 else None


def require(condition, message):
    if not condition:
        raise SystemExit('REFUSED: ' + message)


def current_method(src):
    """The deployed _sanitize method text, or None when absent."""
    services = [node for node in ast.parse(src).body if isinstance(node, ast.ClassDef) and node.name == 'Service']
    require(len(services) == 1, 'expected exactly one Service class')
    methods = [n for n in services[0].body if isinstance(n, ast.FunctionDef) and n.name == '_sanitize']
    require(len(methods) <= 1, 'duplicate sanitizers')
    if not methods:
        return None
    node = methods[0]
    return ''.join(src.splitlines(keepends=True)[node.lineno - 1:node.end_lineno])


def patched(src):
    have = current_method(src)
    if have is not None:
        if have == METHOD:
            require(src.count(NEW_CALL + MARK) == 1, 'current sanitizer not wired before mark_think_literals')
            require(src.count('import re\n') == 1, 'missing or duplicate re import')
            return src
        require(False, 'existing sanitizer differs from v4 (sha256 '
                + hashlib.sha256(have.encode()).hexdigest()[:12] + ')')
    encode = encode_anchor(src)
    require(encode is not None, 'encode_prompt anchor drifted: no known signature in ENCODE_ANCHORS')
    require(src.count(MARK) == 1, 'mark_think_literals anchor drifted')
    src = src.replace(encode, METHOD + '\n' + encode, 1)
    src = src.replace(MARK, NEW_CALL + MARK, 1)
    if 'import re\n' not in src:
        require(src.count('import queue\n') == 1, 'import anchor drifted')
        src = src.replace('import queue\n', 'import queue\nimport re\n', 1)
    compile(src, '<patched server>', 'exec')
    return src


def install(path, content):
    if path.exists() and path.read_text(encoding='utf-8') == content:
        return False
    if path.exists():
        stamp = datetime.now().strftime('%Y%m%d-%H%M%S-%f')
        backup = path.with_name(path.name + '.bak-sanitize-v4-' + stamp)
        shutil.copy2(path, backup)
        print('backup', backup)
    temporary = path.with_name(path.name + '.sanitize-v4-new')
    require(not temporary.exists(), 'temporary output already exists: ' + str(temporary))
    try:
        temporary.write_text(content, encoding='utf-8')
        shutil.copymode(path, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    print('patched', path)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('server', nargs='?', type=Path, default=Path.home() / 'Strata/serve/server.py')
    parser.add_argument('--check', action='store_true', help='verify installed server; never write')
    args = parser.parse_args()
    src = args.server.read_text(encoding='utf-8')
    updated = patched(src)
    if args.check:
        require(updated == src, 'sanitizer is absent or legacy; rerun without --check')
        print('verified current sanitizer', args.server)
        return
    # Calculate and compile every change before touching any file.
    edits = [(args.server, updated)]
    companion = Path(__file__).with_name('test_sanitize.py')
    if companion.exists():
        test_source = companion.read_text(encoding='utf-8')
        compile(test_source, str(companion), 'exec')
        edits.append((args.server.with_name('test_sanitize.py'), test_source))
    changed = False
    for path, content in edits:
        changed = install(path, content) or changed
    if not changed:
        print('already patched (v4)')


if __name__ == '__main__':
    main()
