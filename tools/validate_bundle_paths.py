"""Every notebook_path in databricks.yml resolves to a file on disk.

`databricks bundle validate` already does this, but it needs a workspace to
authenticate against because the bundle expands
`${workspace.current_user.userName}`. That makes it unavailable on a fork, on an
outside pull request, and to anyone who clones the repo without a Databricks
account -- which is most people who will ever read it.

This check needs nothing. It caught the real failure: the silver task pointed at
`./notebooks/03_silver.ipynb` while the file on disk was `02_silver.ipynb`, so
`databricks bundle deploy` -- the command the README tells you to run -- failed.
The notebooks had been renumbered and the bundle had not.

Deliberately a regex rather than a YAML parse: the value wanted is a literal
path, and reading it literally means the check does not need PyYAML and cannot
be fooled by an anchor or a merge key resolving to something else.

    python tools/validate_bundle_paths.py
"""

from __future__ import annotations

import os
import re
import sys

BUNDLE = "databricks.yml"
PATH_PATTERN = re.compile(r"notebook_path:\s*(\S+)")


def main() -> int:
    if not os.path.exists(BUNDLE):
        print(f"{BUNDLE} not found — wrong working directory?")
        return 1

    with open(BUNDLE, encoding="utf-8") as handle:
        text = handle.read()

    references = PATH_PATTERN.findall(text)
    if not references:
        print(f"No notebook_path entries in {BUNDLE}. Expected at least one.")
        return 1

    missing = []
    for reference in references:
        relative = reference.lstrip("./")
        status = "ok" if os.path.exists(relative) else "MISSING"
        if status == "MISSING":
            missing.append(reference)
        print(f"  {status:<8} {reference}")

    print()
    if missing:
        on_disk = sorted(
            f for f in os.listdir("notebooks") if f.endswith(".ipynb")
        ) if os.path.isdir("notebooks") else []
        print(f"{len(missing)} task(s) point at a notebook that does not exist:")
        for reference in missing:
            print(f"  - {reference}")
        print(f"\nnotebooks/ contains: {on_disk}")
        print("`databricks bundle deploy` will fail until these agree.")
        return 1

    print(f"All {len(references)} bundle notebook path(s) resolve.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
