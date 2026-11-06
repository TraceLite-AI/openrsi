"""Represent a change to an existing source file as separately applicable git hunks."""
import difflib
import re


def line_hunks(path, old, new, max_changed_lines):
    if not re.fullmatch(r"[A-Za-z0-9_.\-/]+", path) or ".." in path.split("/"):
        raise ValueError("invalid diff path")
    if not old or not new:
        raise ValueError("bounded hunk mode edits existing nonempty files only")
    lines = list(difflib.unified_diff(old.splitlines(True), new.splitlines(True),
                                      fromfile="a/" + path, tofile="b/" + path, n=3))
    if not lines:
        return []
    headers, chunks = lines[:2], []
    for line in lines[2:]:
        if line.startswith("@@ "):
            chunks.append([])
        chunks[-1].append(line)
    result = []
    for chunk in chunks:
        changed = sum(line.startswith(("+", "-")) for line in chunk[1:])
        if changed > max_changed_lines:
            raise ValueError("hunk exceeds changed-line limit")
        body = "".join(line if line.endswith("\n") else line + "\n\\ No newline at end of file\n" for line in chunk)
        result.append({"op": "diff", "path": path, "unified": "".join(headers) + body})
    return result
